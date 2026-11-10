'use strict';

/**
 * Global simulation constants.
 *
 * Everything in `sim/` is pure JavaScript: no Phaser, no Express, no globals.
 * The same module is used by the HTTP server, the headless runner and the tests.
 */

const TILE_SIZE = 32;
const WORLD = { w: 32, h: 24 };
const WORLD_PX = { w: WORLD.w * TILE_SIZE, h: WORLD.h * TILE_SIZE };

// One tick == 12 simulated minutes, 120 ticks == 24h == one in-game day.
const MINUTES_PER_TICK = 12;
const TICKS_PER_DAY = 120;
const TICK_MS = 500;
const DAY_START_MINUTE = 6 * 60;

const SEASONS = ['spring', 'summer', 'autumn', 'winter'];
const DAYS_PER_SEASON = 4;
const DAYS_PER_YEAR = SEASONS.length * DAYS_PER_SEASON;

/**
 * Landmarks are fixed tiles so that episodes are comparable across runs.
 * `tile` is the authority; pixel positions are always derived from it.
 */
const LANDMARKS = {
  playerStart: { x: 8, y: 4 },
  tank: { x: 3, y: 10 },
  charger: { x: 4, y: 13 },
  silo: { x: 5, y: 4 },
  farmhouse: { x: 2, y: 6 },
};

/** The cultivated field: 8 x 4 = 32 plots. */
const FIELD = { x: 12, y: 8, w: 8, h: 4 };

/** Static obstacles (rocks / fence posts). */
const OBSTACLES = [
  { x: 10, y: 4, type: 'rock' },
  { x: 20, y: 6, type: 'rock' },
  { x: 17, y: 14, type: 'rock' },
  { x: 6, y: 17, type: 'rock' },
  { x: 24, y: 10, type: 'rock' },
  { x: 21, y: 17, type: 'rock' },
];

const MAX_TICKS_PER_ADVANCE = 5000;

const PRESETS = {
  easy: {
    label: 'Easy',
    tick_ms: TICK_MS,
    money: 120,
    seeds: { wheat: 6, rice: 6 },
    can_capacity: 20,
    tank_capacity: 200,
    tank_start: 0.6,
    tank_refill: 0.5, // litres per tick while the pump runs
    water_per_action: 4, // litres per watering (= 0.36 soil moisture, one band)
    weather_mode: 'scripted', // deterministic 4-day cycle
    pest_risk: 0,
    weed_risk: 0,
    market_volatility: 0,
    robots: 0,
    fog_of_war: false,
    growth_speed: 1,
    spoilage: 0,
    objective: 'starter_easy',
  },
  normal: {
    label: 'Normal',
    tick_ms: TICK_MS,
    money: 60,
    seeds: { wheat: 4, rice: 3 },
    can_capacity: 20,
    tank_capacity: 160,
    tank_start: 0.35,
    tank_refill: 0.35,
    water_per_action: 4,
    weather_mode: 'markov',
    pest_risk: 1,
    weed_risk: 1,
    market_volatility: 0.06,
    robots: 2,
    fog_of_war: false,
    // 1.25: a corn cycle takes ~9 real days at 1.0 (water stress, weeds, night,
    // cold), which does not fit a 50-step episode plus planting/harvesting.
    growth_speed: 1.25,
    spoilage: 0.01,
    objective: 'harvest_normal',
  },
  hard: {
    label: 'Hard',
    tick_ms: TICK_MS,
    // 28 = one plot-set of the most valuable crop (4 corn x $7). At $20 the
    // cheapest four-plot plan could not be bought at all: the feasibility probe's
    // own `buy` answered NO_MONEY and planted nothing, and the recorded LLM run
    // could only afford two corn plots - it then missed the $150 target on
    // throughput while still holding 17 kg of unsold produce. $28 keeps the
    // wallet tight (no spare cash for `treat`/`fertilize`) without capping the
    // agent at half a field.
    money: 28,
    seeds: { wheat: 2, rice: 2 },
    can_capacity: 12,
    tank_capacity: 120,
    tank_start: 0.15,
    tank_refill: 0.15,
    water_per_action: 4,
    weather_mode: 'markov_harsh',
    pest_risk: 1.6,
    weed_risk: 1.4,
    market_volatility: 0.12,
    robots: 2,
    fog_of_war: true,
    growth_speed: 1,
    spoilage: 0.02,
    objective: 'profit_hard',
  },
};

/** Weather kinds and their physical effect on the farm. */
const WEATHER = {
  sunny: { label: 'Sunny', temp_offset: 4, humidity: 0.45, wind: 0.35, rain: 0 },
  cloudy: { label: 'Cloudy', temp_offset: 0, humidity: 0.62, wind: 0.25, rain: 0 },
  rain: { label: 'Rain', temp_offset: -4, humidity: 0.9, wind: 0.4, rain: 0.006 },
  storm: { label: 'Storm', temp_offset: -6, humidity: 0.95, wind: 0.85, rain: 0.015 },
  heatwave: { label: 'Heatwave', temp_offset: 9, humidity: 0.3, wind: 0.15, rain: 0 },
  frost: { label: 'Frost', temp_offset: -8, humidity: 0.7, wind: 0.2, rain: 0 },
};

const SEASON_BASE_TEMP = { spring: 16, summer: 27, autumn: 14, winter: 4 };

/** Markov transition weights per season. */
const WEATHER_TRANSITIONS = {
  spring: {
    sunny: { sunny: 0.5, cloudy: 0.3, rain: 0.2 },
    cloudy: { sunny: 0.35, cloudy: 0.3, rain: 0.3, storm: 0.05 },
    rain: { sunny: 0.4, cloudy: 0.35, rain: 0.2, storm: 0.05 },
    storm: { cloudy: 0.6, rain: 0.4 },
    heatwave: { sunny: 0.7, cloudy: 0.3 },
    frost: { cloudy: 0.5, rain: 0.5 },
  },
  summer: {
    sunny: { sunny: 0.6, cloudy: 0.25, heatwave: 0.15 },
    cloudy: { sunny: 0.45, cloudy: 0.25, rain: 0.3 },
    rain: { sunny: 0.6, cloudy: 0.4 },
    storm: { cloudy: 0.5, sunny: 0.5 },
    heatwave: { sunny: 0.5, heatwave: 0.35, storm: 0.15 },
    frost: { cloudy: 1 },
  },
  autumn: {
    sunny: { sunny: 0.35, cloudy: 0.35, rain: 0.3 },
    cloudy: { sunny: 0.25, cloudy: 0.35, rain: 0.4 },
    rain: { sunny: 0.2, cloudy: 0.45, rain: 0.35 },
    storm: { cloudy: 0.5, rain: 0.5 },
    heatwave: { sunny: 0.6, cloudy: 0.4 },
    frost: { cloudy: 0.6, frost: 0.4 },
  },
  winter: {
    sunny: { sunny: 0.3, cloudy: 0.45, frost: 0.25 },
    cloudy: { sunny: 0.15, cloudy: 0.4, rain: 0.15, frost: 0.3 },
    rain: { cloudy: 0.5, rain: 0.2, frost: 0.3 },
    storm: { cloudy: 0.6, rain: 0.4 },
    heatwave: { sunny: 0.5, cloudy: 0.5 },
    frost: { sunny: 0.2, cloudy: 0.4, frost: 0.4 },
  },
};

/** Harsh variant: drier, more extreme weather. */
const WEATHER_TRANSITIONS_HARSH = {
  spring: {
    sunny: { sunny: 0.6, cloudy: 0.2, rain: 0.1, heatwave: 0.1 },
    cloudy: { sunny: 0.5, cloudy: 0.25, rain: 0.2, storm: 0.05 },
    rain: { sunny: 0.5, cloudy: 0.3, storm: 0.2 },
    storm: { cloudy: 0.5, rain: 0.5 },
    heatwave: { sunny: 0.6, heatwave: 0.3, cloudy: 0.1 },
    frost: { cloudy: 0.5, rain: 0.5 },
  },
  summer: {
    sunny: { sunny: 0.5, heatwave: 0.35, cloudy: 0.15 },
    cloudy: { sunny: 0.5, heatwave: 0.2, rain: 0.3 },
    rain: { sunny: 0.5, cloudy: 0.3, storm: 0.2 },
    storm: { cloudy: 0.5, sunny: 0.5 },
    heatwave: { heatwave: 0.55, sunny: 0.35, storm: 0.1 },
    frost: { cloudy: 1 },
  },
  autumn: {
    sunny: { sunny: 0.3, cloudy: 0.3, rain: 0.4 },
    cloudy: { sunny: 0.2, cloudy: 0.3, rain: 0.5 },
    rain: { cloudy: 0.5, rain: 0.4, storm: 0.1 },
    storm: { cloudy: 0.4, rain: 0.6 },
    heatwave: { sunny: 0.6, cloudy: 0.4 },
    frost: { cloudy: 0.4, frost: 0.6 },
  },
  winter: {
    sunny: { sunny: 0.25, cloudy: 0.4, frost: 0.35 },
    cloudy: { sunny: 0.1, cloudy: 0.35, rain: 0.15, frost: 0.4 },
    rain: { cloudy: 0.5, rain: 0.2, frost: 0.3 },
    storm: { cloudy: 0.5, rain: 0.5 },
    heatwave: { sunny: 0.5, cloudy: 0.5 },
    frost: { sunny: 0.15, cloudy: 0.35, frost: 0.5 },
  },
};

/** Deterministic weather cycle for the `easy` preset. */
const SCRIPTED_WEATHER = ['sunny', 'sunny', 'rain', 'sunny', 'cloudy', 'sunny', 'rain', 'sunny'];

const ACTION_REASONS = {
  UNKNOWN_ACTION: 'UNKNOWN_ACTION',
  BAD_ARGS: 'BAD_ARGS',
  PAUSED: 'PAUSED',
  TOO_FAR: 'TOO_FAR',
  OUT_OF_BOUNDS: 'OUT_OF_BOUNDS',
  BLOCKED: 'BLOCKED',
  NO_PLOT: 'NO_PLOT',
  NOT_TILLED: 'NOT_TILLED',
  ALREADY_TILLED: 'ALREADY_TILLED',
  OCCUPIED: 'OCCUPIED',
  NO_CROP: 'NO_CROP',
  CROP_DEAD: 'CROP_DEAD',
  NOT_READY: 'NOT_READY',
  NO_SEEDS: 'NO_SEEDS',
  NO_MONEY: 'NO_MONEY',
  TANK_EMPTY: 'TANK_EMPTY',
  CAN_EMPTY: 'CAN_EMPTY',
  NOT_AT_TANK: 'NOT_AT_TANK',
  NO_ENERGY: 'NO_ENERGY',
  NO_PATH: 'NO_PATH',
  NOTHING_TO_DO: 'NOTHING_TO_DO',
  SILO_FULL: 'SILO_FULL',
  NO_ROBOT: 'NO_ROBOT',
  ROBOT_BUSY: 'ROBOT_BUSY',
};

module.exports = {
  TILE_SIZE,
  WORLD,
  WORLD_PX,
  MINUTES_PER_TICK,
  TICKS_PER_DAY,
  TICK_MS,
  DAY_START_MINUTE,
  SEASONS,
  DAYS_PER_SEASON,
  DAYS_PER_YEAR,
  LANDMARKS,
  FIELD,
  OBSTACLES,
  MAX_TICKS_PER_ADVANCE,
  PRESETS,
  WEATHER,
  SEASON_BASE_TEMP,
  WEATHER_TRANSITIONS,
  WEATHER_TRANSITIONS_HARSH,
  SCRIPTED_WEATHER,
  ACTION_REASONS,
};
