'use strict';

const { TILE_SIZE } = require('./config');
const { round } = require('./util');
const { clockString, isNight, elapsedMinutes } = require('./systems/time');
const soilSystem = require('./systems/soil');
const cropSystem = require('./systems/crop');
const economy = require('./systems/economy');
const { recentEvents } = require('./systems/events');
const { hashString } = require('./rng');
const { computeScore } = require('./metrics');

const DEFAULT_FOG_RADIUS = 10;

function plotView(world, plot, hidden) {
  const player = { x: world.player.tile[0], y: world.player.tile[1] };
  const base = {
    id: plot.id,
    tile: plot.tile,
    x: plot.tile[0],
    y: plot.tile[1],
    row: plot.row,
    col: plot.col,
    distance: Math.abs(plot.tile[0] - player.x) + Math.abs(plot.tile[1] - player.y),
  };
  if (hidden) return { ...base, known: false };

  const def = plot.crop ? cropSystem.defFor(plot.crop.type) : null;
  const crop = plot.crop
    ? {
        type: plot.crop.type,
        label: def.label,
        growth: round(plot.crop.growth, 3),
        stage: cropSystem.stageName(plot.crop),
        stage_index: cropSystem.stageIndex(plot.crop),
        ready: plot.crop.ready,
        dead: plot.crop.dead,
        health: round(plot.crop.health, 3),
        quality: round(plot.crop.quality, 3),
        weeds: round(plot.crop.weeds, 2),
        disease: plot.crop.disease ? plot.crop.disease.type : null,
        disease_severity: plot.crop.disease ? round(plot.crop.disease.severity, 2) : 0,
        days_to_ripe: round(def.days * (1 - plot.crop.growth), 2),
        planted_day: plot.crop.planted_day,
      }
    : null;

  return {
    ...base,
    known: true,
    tilled: plot.tilled,
    moisture: round(plot.soil.moisture, 3),
    fertility: round(plot.soil.fertility, 3),
    band: soilSystem.moistureBand(plot, def),
    needs_water: crop ? soilSystem.needsWater(plot, def) : plot.soil.moisture < 0.35,
    crop,
  };
}

function catalogView(world) {
  const catalog = {};
  for (const [type, def] of Object.entries(world.catalog)) {
    catalog[type] = {
      label: def.label,
      days: def.days,
      optimal: def.optimal,
      seed_cost: def.seed_cost,
      sell_price: def.sell_price,
      yield_kg: def.yield_kg,
      color: def.color,
      seasons: def.seasons,
      plant_energy: def.plant_energy,
      harvest_energy: def.harvest_energy,
    };
  }
  return catalog;
}

/**
 * Full observable state. This is what the renderer draws and what an agent
 * reads; it never exposes the RNG stream.
 */
function observe(world, options = {}) {
  const eventCount = options.events === undefined ? 12 : options.events;
  const fogRadius = options.fog_radius || DEFAULT_FOG_RADIUS;
  const fog = world.presetConfig.fog_of_war && options.fog !== false;
  const playerTile = { x: world.player.tile[0], y: world.player.tile[1] };

  const plots = world.plots.map((plot) => {
    const hidden =
      fog &&
      Math.abs(plot.tile[0] - playerTile.x) + Math.abs(plot.tile[1] - playerTile.y) > fogRadius;
    return plotView(world, plot, hidden);
  });

  return {
    schema_version: world.schema_version,
    tick: world.tick,
    tick_ms: world.tick_ms,
    seed: world.seed,
    preset: world.preset,
    preset_label: world.presetConfig.label,
    task: world.task_text,
    time: {
      day: world.time.day,
      minute: world.time.minute,
      clock: clockString(world),
      season: world.time.season,
      season_day: world.time.season_day,
      year: world.time.year,
      paused: world.time.paused,
      speed: world.time.speed,
      is_night: isNight(world),
      elapsed_minutes: elapsedMinutes(world),
    },
    weather: {
      type: world.weather.type,
      label: world.weather.label,
      temp_c: world.weather.temp_c,
      humidity: world.weather.humidity,
      wind: world.weather.wind,
      rain_rate: world.weather.rain_rate,
      rain_today: world.weather.rain_today,
      frost: world.weather.frost,
      forecast: world.weather.forecast,
    },
    world: {
      w: world.world.w,
      h: world.world.h,
      tile_size: TILE_SIZE,
      px_w: world.world.w * TILE_SIZE,
      px_h: world.world.h * TILE_SIZE,
    },
    player: {
      tile: [playerTile.x, playerTile.y],
      pos: [playerTile.x * TILE_SIZE + TILE_SIZE / 2, playerTile.y * TILE_SIZE + TILE_SIZE / 2],
      facing: world.player.facing,
      energy: round(world.player.energy, 1),
      max_energy: world.player.max_energy,
      money: world.player.money,
      can: { litres: round(world.player.can.liters, 2), capacity: world.player.can.capacity },
      seeds: { ...world.player.seeds },
    },
    tank: {
      tile: world.tank.tile,
      pos: [
        world.tank.tile[0] * TILE_SIZE + TILE_SIZE / 2,
        world.tank.tile[1] * TILE_SIZE + TILE_SIZE / 2,
      ],
      litres: round(world.tank.liters, 2),
      capacity: world.tank.capacity,
      pct: round(world.tank.liters / world.tank.capacity, 3),
      pump_active: Boolean(world.tank.pump_active),
    },
    plots,
    obstacles: world.obstacles,
    landmarks: world.landmarks,
    robots: world.robots.map((robot) => ({
      id: robot.id,
      tile: robot.tile,
      pos: [robot.tile[0] * TILE_SIZE + TILE_SIZE / 2, robot.tile[1] * TILE_SIZE + TILE_SIZE / 2],
      battery: round(robot.battery, 3),
      task: robot.task,
      status: robot.status,
      target: robot.target,
      tank_litres: round(robot.tank.liters, 2),
      watered: robot.watered,
    })),
    market: {
      prices: Object.fromEntries(
        Object.keys(world.catalog).map((type) => [type, economy.priceFor(world, type)]),
      ),
      volatility: world.market.volatility || 0,
      contracts: world.market.contracts,
    },
    silo: {
      capacity_kg: world.silo.capacity_kg,
      total_kg: round(economy.totalSiloKg(world), 2),
      items: Object.fromEntries(
        Object.entries(world.silo.items).map(([type, item]) => [
          type,
          { kg: round(item.kg, 2), quality: round(item.quality, 2) },
        ]),
      ),
    },
    objective: {
      id: world.objective.id,
      label: world.objective.label,
      text: world.objective.text,
      deadline_day: world.objective.deadline_day,
      reward: world.objective.reward,
      progress: world.objective.progress,
      is_complete: world.objective.is_complete,
      is_failed: world.objective.is_failed,
    },
    metrics: { ...world.metrics, score: computeScore(world) },
    events: eventCount > 0 ? recentEvents(world, eventCount) : [],
    action_log: world.action_log.slice(-10),
    catalog: catalogView(world),
  };
}

/** Compact, order-stable digest used by the determinism tests. */
function stateDigest(world) {
  const payload = {
    tick: world.tick,
    day: world.time.day,
    minute: world.time.minute,
    weather: world.weather.type,
    temp: world.weather.temp_c,
    player: world.player.tile,
    energy: round(world.player.energy, 2),
    money: world.player.money,
    tank: round(world.tank.liters, 2),
    can: round(world.player.can.liters, 2),
    seeds: world.player.seeds,
    plots: world.plots.map((plot) => [
      plot.id,
      plot.tilled,
      round(plot.soil.moisture, 3),
      plot.crop
        ? [
            plot.crop.type,
            round(plot.crop.growth, 3),
            round(plot.crop.health, 3),
            plot.crop.ready,
            plot.crop.dead,
            round(plot.crop.weeds, 2),
          ]
        : null,
    ]),
    silo: Object.fromEntries(
      Object.entries(world.silo.items).map(([type, item]) => [type, round(item.kg, 2)]),
    ),
    metrics: {
      harvests: world.metrics.harvests,
      kg: round(world.metrics.kg_harvested, 2),
      earned: world.metrics.earned,
      water_used: world.metrics.water_used,
    },
    objective: [world.objective.is_complete, world.objective.is_failed],
  };
  return hashString(JSON.stringify(payload));
}

module.exports = { observe, plotView, catalogView, stateDigest, DEFAULT_FOG_RADIUS };
