'use strict';

/**
 * Legacy compatibility layer for the original `get_game_state` /
 * `post_game_state` protocol.
 *
 * The old Python agents work on a 25 px cell grid (their `move` tool only
 * accepts +-25 px). The new simulation uses 32 px tiles, so the adapter maps
 * cell <-> tile 1:1 and converts pixels: `legacy_px = tile * 25`.
 *
 * The old landmarks were chosen to line up exactly:
 *   player start (8,4)  -> [200,100]
 *   water tank   (3,10) -> [75,250]
 *   rock         (10,4) -> [250,100]
 *   field        (12..19, 8..11) -> [300,200] .. [475,275]
 */

const { WORLD } = require('./config');
const { round } = require('./util');
const { clockString } = require('./systems/time');
const soilSystem = require('./systems/soil');
const { irrigate } = require('./systems/soil');
const cropSystem = require('./systems/crop');
const { fillCan, takeFromCan } = require('./systems/water');
const { pushEvent } = require('./systems/events');
const { plotAt, isObstacle, inBounds } = require('./world');

const LEGACY_CELL = 25;

const toLegacy = (tile) => [tile[0] * LEGACY_CELL, tile[1] * LEGACY_CELL];
const toTile = (px, py) => ({
  x: Math.round(Number(px) / LEGACY_CELL),
  y: Math.round(Number(py) / LEGACY_CELL),
});

function legacyView(world) {
  const crops = {};
  world.plots.forEach((plot, index) => {
    const def = plot.crop ? cropSystem.defFor(plot.crop.type) : null;
    crops[`crop${index + 1}`] = {
      pos: toLegacy(plot.tile),
      name: def ? def.label.toLowerCase() : '',
      planted: Boolean(plot.crop),
      needs_water: plot.crop ? soilSystem.needsWater(plot, def) : true,
      ready: Boolean(plot.crop && plot.crop.ready),
      dead: Boolean(plot.crop && plot.crop.dead),
      growth: plot.crop ? round(plot.crop.growth, 3) : 0,
      moisture: round(plot.soil.moisture, 3),
      plot_id: plot.id,
    };
  });

  return {
    // ---- legacy fields (unchanged shape) ----
    grid_size: [WORLD.w * LEGACY_CELL, WORLD.h * LEGACY_CELL],
    player_pos: toLegacy(world.player.tile),
    crops,
    obstacles: world.obstacles.map((o) => toLegacy(o.tile)),
    water_available: world.player.can.liters > 0,
    water_tank: toLegacy(world.tank.tile),
    goal_completed: Boolean(world.objective.is_complete),
    task: world.task_text,
    // ---- extra context for newer agents; harmless for the old ones ----
    tick: world.tick,
    day: world.time.day,
    minute: world.time.minute,
    clock: clockString(world),
    season: world.time.season,
    weather: world.weather.type,
    weather_label: world.weather.label,
    tank_liters: round(world.tank.liters, 2),
    can_liters: round(world.player.can.liters, 2),
    money: world.player.money,
    seed_can: world.player.can.capacity,
    objective: {
      id: world.objective.id,
      text: world.objective.text,
      progress: world.objective.progress,
      is_complete: world.objective.is_complete,
      is_failed: world.objective.is_failed,
    },
    preset: world.preset,
    seed: world.seed,
  };
}

function movePlayer(world, tile) {
  if (!inBounds(tile.x, tile.y)) return 'OUT_OF_BOUNDS';
  if (isObstacle(world, tile.x, tile.y)) return 'BLOCKED';
  world.player.tile = [tile.x, tile.y];
  return null;
}

/**
 * Apply a legacy `post_game_state` body.
 * Only the transitions the old protocol can express are replayed:
 * player position, planting, watering and "collect water".
 */
function applyLegacyPost(world, body = {}) {
  const applied = [];
  let rejected = null;

  if (Array.isArray(body.player_pos) && body.player_pos.length >= 2) {
    const tile = toTile(body.player_pos[0], body.player_pos[1]);
    const error = movePlayer(world, tile);
    if (error) rejected = { reason: error, target: body.player_pos };
    else applied.push({ type: 'player_pos', tile: [tile.x, tile.y] });
  }

  if (body.water_available === true && world.player.can.liters <= 0.01) {
    const litres = fillCan(world, world.player.can.capacity);
    if (litres > 0) applied.push({ type: 'collect_water', litres: round(litres, 2) });
  }

  if (body.crops && typeof body.crops === 'object') {
    for (const [key, entry] of Object.entries(body.crops)) {
      if (!entry || !Array.isArray(entry.pos)) continue;
      const tile = toTile(entry.pos[0], entry.pos[1]);
      const plot = plotAt(world, tile.x, tile.y);
      if (!plot) {
        rejected = rejected || { reason: 'NO_PLOT', crop: key, tile: [tile.x, tile.y] };
        continue;
      }

      if (entry.planted === true && !plot.crop) {
        const type = String(entry.name || 'wheat').toLowerCase();
        const def = cropSystem.defFor(type);
        if (!def) {
          rejected = rejected || { reason: 'BAD_ARGS', crop: type };
          continue;
        }
        // The old agents never tilled, so the adapter tills implicitly.
        plot.tilled = true;
        plot.crop = cropSystem.makeCrop(world, type);
        pushEvent(world, 'planted', `Legacy plant: ${def.label} on ${plot.id}.`, {
          plot: plot.id,
          crop: type,
          legacy: true,
        });
        applied.push({ type: 'plant', plot: plot.id, crop: type });
      }

      if (entry.planted === false && plot.crop && plot.crop.growth < 0.999) {
        cropSystem.clear(world, plot);
        applied.push({ type: 'clear', plot: plot.id });
      }

      if (entry.needs_water === false && plot.crop) {
        const def = cropSystem.defFor(plot.crop.type);
        const cost = world.presetConfig.water_per_action;
        if (!soilSystem.needsWater(plot, def)) continue; // already wet enough

        if (world.player.can.liters + 1e-9 < cost) {
          const litres = fillCan(world, cost);
          if (litres + 1e-9 < cost) {
            rejected = rejected || { reason: 'CAN_EMPTY', plot: plot.id };
            continue;
          }
        }
        takeFromCan(world, cost);
        irrigate(plot, cost, world);
        pushEvent(world, 'watered', `Legacy watering on ${plot.id}.`, {
          plot: plot.id,
          legacy: true,
        });
        applied.push({ type: 'water', plot: plot.id, litres: cost });
      }
    }
  }

  if (typeof body.task === 'string' && body.task.trim()) {
    world.task_text = body.task.trim();
    applied.push({ type: 'task', text: world.task_text });
  }

  return { applied, rejected, state: legacyView(world) };
}

function setTask(world, text) {
  world.task_text = String(text || '').trim() || world.objective.text;
  return world.task_text;
}

module.exports = { LEGACY_CELL, legacyView, applyLegacyPost, setTask, toLegacy, toTile };
