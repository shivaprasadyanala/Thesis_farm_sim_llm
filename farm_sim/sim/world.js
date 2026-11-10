'use strict';

const {
  TILE_SIZE,
  WORLD,
  FIELD,
  OBSTACLES,
  LANDMARKS,
  PRESETS,
  DAY_START_MINUTE,
  TICK_MS,
} = require('./config');
const { Rng } = require('./rng');
const { clamp, round } = require('./util');
const CROPS = require('./data/crops.json');

const weather = require('./systems/weather');
const economy = require('./systems/economy');
const robots = require('./systems/robots');
const objective = require('./systems/objective');
const { pushEvent } = require('./systems/events');

const SCHEMA_VERSION = 3;

function tileCenter(tile) {
  return { x: tile[0] * TILE_SIZE + TILE_SIZE / 2, y: tile[1] * TILE_SIZE + TILE_SIZE / 2 };
}

function makePlots(world, presetConfig) {
  const plots = [];
  let index = 0;
  for (let row = 0; row < FIELD.h; row += 1) {
    for (let col = 0; col < FIELD.w; col += 1) {
      index += 1;
      const x = FIELD.x + col;
      const y = FIELD.y + row;
      const base = presetConfig.tank_start;
      plots.push({
        id: `p${String(index).padStart(2, '0')}`,
        tile: [x, y],
        row,
        col,
        tilled: false,
        crop: null,
        cleared_tick: 0,
        soil: {
          moisture: round(clamp(world.rng.range(base - 0.1, base + 0.12), 0.05, 0.95), 3),
          fertility: round(world.rng.range(0.75, 0.95), 3),
          ph: 6.5,
          temp_c: 18,
          wet_ticks: 0,
          // -1 = never irrigated. A plain `0` cannot be told apart from "watered
          // on tick 0", which made freshly planted plots count as watered for
          // `countWatered` (and the easy objective complete on planting alone).
          last_watered_tick: -1,
          et: 0,
        },
      });
    }
  }
  return plots;
}

function createMetrics() {
  return {
    actions: 0,
    valid_actions: 0,
    invalid_actions: 0,
    harvests: 0,
    kg_harvested: 0,
    kg_sold: 0,
    kg_spoiled: 0,
    earned: 0,
    spent: 0,
    water_used: 0,
    water_collected: 0,
    objective_completed: 0,
    robot_tiles: 0,
    crops_died: 0,
    reward: 0,
  };
}

/**
 * Build a fresh world. Deterministic for a given (seed, preset, task).
 */
function createWorld(options = {}) {
  const presetName = PRESETS[options.preset] ? options.preset : 'normal';
  const presetConfig = { ...PRESETS[presetName] };
  const rng = new Rng(options.seed === undefined ? 1 : options.seed);

  const world = {
    schema_version: SCHEMA_VERSION,
    seed: rng.seed,
    preset: presetName,
    presetConfig,
    catalog: CROPS,
    tick: 0,
    tick_ms: presetConfig.tick_ms || TICK_MS,
    time: {
      day: 1,
      minute: DAY_START_MINUTE,
      start_minute: DAY_START_MINUTE,
      season: 'spring',
      season_day: 1,
      year: 1,
      paused: false,
      speed: 1,
    },
    weather: {},
    player: {
      tile: [LANDMARKS.playerStart.x, LANDMARKS.playerStart.y],
      facing: 'down',
      energy: 100,
      max_energy: 100,
      money: presetConfig.money,
      seeds: { ...presetConfig.seeds },
      can: { liters: 0, capacity: presetConfig.can_capacity },
      resting: false,
      moved_this_tick: false,
    },
    tank: {
      tile: [LANDMARKS.tank.x, LANDMARKS.tank.y],
      liters: round(presetConfig.tank_capacity * presetConfig.tank_start, 1),
      capacity: presetConfig.tank_capacity,
      pump_active: true,
      last_gain: 0,
    },
    plots: [],
    obstacles: OBSTACLES.map((o) => ({
      tile: [o.x, o.y],
      pos: [o.x * TILE_SIZE + TILE_SIZE / 2, o.y * TILE_SIZE + TILE_SIZE / 2],
      type: o.type,
    })),
    landmarks: {
      tank: [LANDMARKS.tank.x, LANDMARKS.tank.y],
      charger: [LANDMARKS.charger.x, LANDMARKS.charger.y],
      silo: [LANDMARKS.silo.x, LANDMARKS.silo.y],
      farmhouse: [LANDMARKS.farmhouse.x, LANDMARKS.farmhouse.y],
    },
    world: { w: WORLD.w, h: WORLD.h, tile_size: TILE_SIZE },
    robots: [],
    market: {},
    silo: {},
    objective: {},
    events: [],
    metrics: createMetrics(),
    action_log: [],
    rng,
    task_text: options.taskText || null,
  };

  world.plots = makePlots(world, presetConfig);

  weather.init(world);
  economy.init(world);
  robots.init(world);
  objective.init(world, options.objective || presetConfig.objective);
  if (!world.task_text) world.task_text = world.objective.text;

  pushEvent(world, 'session_start', `Farm session started (${presetName}, seed ${world.seed}).`, {
    preset: presetName,
    seed: world.seed,
  });

  return world;
}

/** O(1) plot lookup; the index lives outside the serialisable world. */
const plotIndexCache = new WeakMap();
function plotIndex(world) {
  let index = plotIndexCache.get(world);
  if (!index) {
    index = new Map(world.plots.map((plot) => [`${plot.tile[0]},${plot.tile[1]}`, plot]));
    plotIndexCache.set(world, index);
  }
  return index;
}

function plotAt(world, x, y) {
  return plotIndex(world).get(`${x},${y}`) || null;
}

function isObstacle(world, x, y) {
  return world.obstacles.some((o) => o.tile[0] === x && o.tile[1] === y);
}

function inBounds(x, y) {
  return x >= 0 && y >= 0 && x < WORLD.w && y < WORLD.h;
}

/** Reset per-tick counters (called by the loop before systems run). */
function beginTick(world) {
  world.player.resting = false;
}

module.exports = {
  SCHEMA_VERSION,
  createWorld,
  createMetrics,
  plotIndex,
  plotAt,
  isObstacle,
  inBounds,
  tileCenter,
  beginTick,
};
