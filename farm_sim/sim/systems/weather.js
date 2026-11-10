'use strict';

const {
  WEATHER,
  SEASON_BASE_TEMP,
  WEATHER_TRANSITIONS,
  WEATHER_TRANSITIONS_HARSH,
  SCRIPTED_WEATHER,
  TICKS_PER_DAY,
} = require('../config');
const { seasonForDay } = require('./time');
const { pushEvent } = require('./events');
const { clamp01, round } = require('../util');

function transitionTable(world) {
  return world.presetConfig.weather_mode === 'markov_harsh'
    ? WEATHER_TRANSITIONS_HARSH
    : WEATHER_TRANSITIONS;
}

/** Draw the next day's weather. Deterministic given the world RNG. */
function pickNextType(world) {
  const mode = world.presetConfig.weather_mode;
  if (mode === 'scripted') {
    return SCRIPTED_WEATHER[(world.time.day - 1) % SCRIPTED_WEATHER.length];
  }
  const table = transitionTable(world);
  const seasonTable = table[world.time.season] || table.spring;
  const weights = seasonTable[world.weather.type] || seasonTable.sunny;
  return world.rng.weighted(weights);
}

/**
 * Naive forecast: follows the most likely transition each day and never draws
 * from the RNG (so looking at the forecast cannot change the episode).
 */
function buildForecast(world, days = 3) {
  const mode = world.presetConfig.weather_mode;
  const table = transitionTable(world);
  const out = [];
  let type = world.weather.type;
  for (let i = 1; i <= days; i += 1) {
    const day = world.time.day + i;
    const season = seasonForDay(day);
    if (mode === 'scripted') {
      type = SCRIPTED_WEATHER[(day - 1) % SCRIPTED_WEATHER.length];
    } else {
      const weights = (table[season] || table.spring)[type] || { sunny: 1 };
      type = Object.entries(weights).sort((a, b) => b[1] - a[1])[0][0];
    }
    out.push({ day, season, type, label: WEATHER[type].label });
  }
  return out;
}

/** Physical weather parameters for the current tick. */
function applyPhysics(world) {
  const w = world.weather;
  const def = WEATHER[w.type] || WEATHER.sunny;
  const base = SEASON_BASE_TEMP[world.time.season] ?? 16;
  // Diurnal curve: coldest at 00:00, warmest around 12:00-14:00.
  const diurnal = 5 * Math.sin((((world.time.minute - 360) + 1440) % 1440) / 1440 * 2 * Math.PI);

  w.label = def.label;
  w.temp_c = round(base + def.temp_offset + diurnal, 1);
  w.humidity = round(clamp01(def.humidity + (w.type === 'rain' ? 0.03 : 0)), 3);
  w.wind = def.wind;
  w.rain_rate = def.rain;
  w.rain_today = round((w.rain_today || 0) + def.rain * 100, 2); // mm-equivalent
  w.frost = w.temp_c <= 1.5;
  w.growth_light = w.type === 'sunny' ? 1 : w.type === 'cloudy' ? 0.75 : 0.45;
}

function update(world, clock) {
  const w = world.weather;
  const newDay = Boolean(clock && clock.newDay);

  if (newDay || !w.type) {
    const previous = w.type;
    const next = pickNextType(world);
    w.type = next;
    w.since_tick = world.tick;
    w.rain_today = 0;
    w.forecast = buildForecast(world);
    if (previous && previous !== next) {
      pushEvent(world, 'weather', `Weather turned to ${WEATHER[next].label.toLowerCase()}.`, {
        from: previous,
        to: next,
      });
    }
    if (newDay && next === 'frost') {
      pushEvent(world, 'weather_alert', 'Frost overnight: tender crops will suffer.', {});
    }
    if (newDay && next === 'heatwave') {
      pushEvent(world, 'weather_alert', 'Heatwave: soil will dry out much faster today.', {});
    }
  }

  applyPhysics(world);
  return w;
}

/** Called once when a world is created. */
function init(world) {
  world.weather.type = world.presetConfig.weather_mode === 'scripted'
    ? SCRIPTED_WEATHER[0]
    : 'sunny';
  world.weather.rain_today = 0;
  applyPhysics(world);
  world.weather.forecast = buildForecast(world);
  world.weather.since_tick = 0;
  return world.weather;
}

module.exports = { init, update, applyPhysics, buildForecast, pickNextType, TICKS_PER_DAY };
