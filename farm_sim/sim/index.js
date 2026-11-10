'use strict';

/**
 * Simulation facade.
 *
 * One instance owns one world. The HTTP server keeps a single instance and
 * ticks it on a timer; the headless runner and the tests drive it directly.
 */

const {
  PRESETS,
  TILE_SIZE,
  WORLD,
  WORLD_PX,
  MINUTES_PER_TICK,
  TICKS_PER_DAY,
  DAYS_PER_SEASON,
  TICK_MS,
  MAX_TICKS_PER_ADVANCE,
  ACTION_REASONS,
} = require('./config');
const { createWorld, SCHEMA_VERSION } = require('./world');
const { tickWorld, advanceTicks } = require('./loop');
const { applyAction, ACTION_NAMES } = require('./actions');
const { observe, stateDigest, catalogView } = require('./observation');
const legacy = require('./legacy');
const { computeScore } = require('./metrics');
const { TOOLS } = require('./tools');
const { CROPS } = require('./systems/crop');
const objective = require('./systems/objective');

const VERSION = '1.0.0';

class Simulation {
  constructor(options = {}) {
    this.defaults = {
      seed: options.seed === undefined ? 1 : options.seed,
      preset: options.preset || 'normal',
      taskText: options.taskText || null,
      objective: options.objective || null,
    };
    this.episode = 0;
    this.createdAt = Date.now();
    this.world = null;
    this.reset(this.defaults);
  }

  /** Start a new episode. Returns the fresh observation. */
  reset(options = {}) {
    const seed = options.seed !== undefined ? options.seed : this.defaults.seed;
    const preset = PRESETS[options.preset] ? options.preset : this.defaults.preset;
    this.world = createWorld({
      seed,
      preset,
      taskText: options.taskText !== undefined ? options.taskText : this.defaults.taskText,
      objective: options.objective || this.defaults.objective,
    });
    this.episode += 1;
    this.world.episode = this.episode;
    if (options.paused !== undefined) this.world.time.paused = Boolean(options.paused);
    if (Number.isFinite(Number(options.speed)) && Number(options.speed) > 0) {
      this.world.time.speed = Number(options.speed);
    }
    return this.observe();
  }

  tick() {
    return tickWorld(this.world);
  }

  /**
   * Fast-forward without acting.
   * @param {{ticks?: number, minutes?: number}} options
   */
  advance(options = {}) {
    let ticks = Number(options.ticks);
    if (!Number.isFinite(ticks)) {
      ticks = Math.floor(Number(options.minutes || 0) / MINUTES_PER_TICK);
    }
    ticks = Math.max(0, Math.min(Math.floor(ticks), MAX_TICKS_PER_ADVANCE));
    const result = advanceTicks(this.world, ticks);
    return {
      ...result,
      ticks_requested: ticks,
      clock: this.observe({ events: 0 }).time.clock,
      score: computeScore(this.world),
    };
  }

  /** Apply one action. `advance` is handled here because it drives the loop. */
  act(input) {
    const payload = typeof input === 'string' ? { type: input } : input || {};
    const type = String(payload.type || payload.action || '').trim().toLowerCase();

    if (type === 'advance') {
      const result = this.advance(payload);
      return {
        ok: true,
        action: 'advance',
        reason: null,
        message: `Advanced ${result.ticks} tick(s).`,
        data: {
          ticks: result.ticks,
          days: result.days,
          clock: result.clock,
          score: result.score,
          events: result.events.map((e) => e.message),
        },
        reward: 0,
      };
    }

    return applyAction(this.world, payload);
  }

  observe(options = {}) {
    return observe(this.world, options);
  }

  digest() {
    return stateDigest(this.world);
  }

  legacy() {
    return legacy.legacyView(this.world);
  }

  legacyPost(body) {
    return legacy.applyLegacyPost(this.world, body);
  }

  setTask(text) {
    return legacy.setTask(this.world, text);
  }

  setPaused(paused) {
    this.world.time.paused = Boolean(paused);
    return this.world.time.paused;
  }

  setSpeed(speed) {
    const value = Number(speed);
    if (!Number.isFinite(value) || value <= 0) return this.world.time.speed;
    this.world.time.speed = Math.min(20, value);
    return this.world.time.speed;
  }

  configView() {
    const presets = {};
    for (const [name, preset] of Object.entries(PRESETS)) {
      presets[name] = {
        label: preset.label,
        tick_ms: preset.tick_ms,
        robots: preset.robots,
        fog_of_war: preset.fog_of_war,
        can_capacity: preset.can_capacity,
        tank_capacity: preset.tank_capacity,
        // Litres one `water` pours. Published because an agent has to budget
        // waterings before it can watch a `water` result: the prompt used to
        // hard-code "2 L per watering" and was wrong after the rebalance to 4 L.
        water_per_action: preset.water_per_action,
        starting_money: preset.money,
        starting_seeds: preset.seeds,
        objective: preset.objective,
      };
    }
    return {
      name: 'farm-sim',
      version: VERSION,
      schema_version: SCHEMA_VERSION,
      tick_ms: TICK_MS,
      minutes_per_tick: MINUTES_PER_TICK,
      ticks_per_day: TICKS_PER_DAY,
      // Season length, published for the same reason as `water_per_action`: an
      // agent planning a 5-day corn cycle against an 18-day deadline has to know
      // how long a season lasts and which one it is standing in, otherwise the
      // winter stall on the hard preset arrives unannounced.
      days_per_season: DAYS_PER_SEASON,
      tile_size: TILE_SIZE,
      world: { w: WORLD.w, h: WORLD.h, px_w: WORLD_PX.w, px_h: WORLD_PX.h },
      current: { preset: this.world.preset, seed: this.world.seed, episode: this.episode },
      presets,
      actions: ACTION_NAMES,
      reasons: ACTION_REASONS,
      crops: catalogView(this.world),
      objectives: objective.catalog(),
    };
  }

  /**
   * The objectives a reset may select, independent of the preset.
   *
   * `POST /api/v1/reset {preset, objective}` pairs any world with any goal, so
   * the difficulty of the environment and the task itself can be varied
   * separately (e.g. `preset: 'easy'` + `objective: 'harvest_normal'`).
   */
  objectivesView() {
    return { objectives: objective.catalog() };
  }

  schemaView() {
    return {
      name: 'farm-sim',
      version: VERSION,
      description:
        'Server-authoritative farm simulation. Ticks advance on their own; every action returns {ok, reason, message, data}.',
      tools: TOOLS,
    };
  }
}

module.exports = {
  Simulation,
  VERSION,
  SCHEMA_VERSION,
  PRESETS,
  CROPS,
  TOOLS,
  ACTION_NAMES,
  OBJECTIVES: objective.OBJECTIVES,
  observe,
  stateDigest,
};
