'use strict';

const { isNight } = require('./time');
const { clamp, clamp01, round } = require('../util');

/**
 * Litres needed to raise a plot's moisture by 1.0.
 *
 * Chosen so that one watering (`water_per_action` = 4 L) lifts a plot by `0.36`
 * - about one crop band width (`optimal` spans 0.35 for every crop), i.e. from
 * the dry edge straight to the wet edge and no further. A larger bump overshoots
 * the band and drowns the crop, a smaller one needs more waterings per day than
 * an episode budget allows.
 */
const MOISTURE_PER_LITRE = 0.09;
/** Reference evapotranspiration at moderate conditions (moisture per tick). */
const BASE_ET = 0.0035;
/** Surface runoff above field capacity. */
const DRAINAGE = 0.004;

/**
 * Reference evapotranspiration: hotter, windier and sunnier days dry the soil
 * faster; a wet surface also evaporates faster than a dry one.
 */
function evapotranspiration(world, plot) {
  const w = world.weather;
  // Capped at 1.6x: an uncapped multiplier turned a heatwave into ~6x reference
  // ET (~0.7 moisture/day), which needs two waterings per plot per day - more
  // labour than a step-budgeted agent can supply, so one hot day killed a farm.
  const tempFactor = clamp((w.temp_c + 4) / 24, 0.05, 1.6);
  const windFactor = 1 + w.wind * 0.5;
  const lightFactor = isNight(world) ? 0.25 : 1;
  const surfaceFactor = 1 - 0.45 * clamp01(plot.soil.moisture);
  return BASE_ET * tempFactor * windFactor * lightFactor * surfaceFactor;
}

/** Moisture added by irrigation (litres) - the only water sink is the tank. */
function moistureFromLitres(litres) {
  return litres * MOISTURE_PER_LITRE;
}

function irrigate(plot, litres, world) {
  plot.soil.moisture = clamp01(plot.soil.moisture + moistureFromLitres(litres));
  if (world) plot.soil.last_watered_tick = world.tick;
  return plot.soil.moisture;
}

function update(world, plot) {
  const s = plot.soil;
  const et = evapotranspiration(world, plot);

  s.moisture -= et;
  if (world.weather.rain_rate > 0) s.moisture += world.weather.rain_rate;
  if (s.moisture > 0.92) s.moisture -= DRAINAGE;
  s.moisture = clamp01(s.moisture);

  // Soil temperature follows air temperature with a lag.
  s.temp_c = round(s.temp_c + (world.weather.temp_c - s.temp_c) * 0.02, 1);

  // Growing crops mine nutrients; fallow tilled soil slowly recovers.
  if (plot.crop && !plot.crop.dead) s.fertility -= 0.000012;
  else if (plot.tilled) s.fertility += 0.000004;
  s.fertility = clamp01(s.fertility);

  s.wet_ticks = s.moisture > 0.15 ? s.wet_ticks + 1 : 0;
  s.et = round(et, 5);
  return et;
}

function needsWater(plot, def) {
  const lo = def ? def.optimal[0] : 0.4;
  return plot.soil.moisture < lo;
}

function moistureBand(plot, def) {
  const [lo, hi] = def ? def.optimal : [0.4, 0.8];
  if (plot.soil.moisture < lo * 0.5) return 'critical';
  if (plot.soil.moisture < lo) return 'dry';
  if (plot.soil.moisture > hi) return 'soggy';
  return 'good';
}

module.exports = {
  MOISTURE_PER_LITRE,
  BASE_ET,
  DRAINAGE,
  evapotranspiration,
  moistureFromLitres,
  irrigate,
  update,
  needsWater,
  moistureBand,
};
