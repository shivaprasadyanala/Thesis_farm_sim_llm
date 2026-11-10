'use strict';

const { MINUTES_PER_TICK, SEASONS, DAYS_PER_SEASON, DAYS_PER_YEAR } = require('../config');

const MINUTES_PER_DAY = 24 * 60;

function seasonForDay(day) {
  const index = Math.floor((((day - 1) % DAYS_PER_YEAR) + DAYS_PER_YEAR) % DAYS_PER_YEAR / DAYS_PER_SEASON);
  return SEASONS[index];
}

function seasonDay(day) {
  return ((((day - 1) % DAYS_PER_SEASON) + DAYS_PER_SEASON) % DAYS_PER_SEASON) + 1;
}

/**
 * Advance the world clock by exactly one tick.
 * @returns {{newDay: boolean, newSeason: boolean, previousSeason: string}}
 */
function advanceClock(world) {
  const t = world.time;
  const previousSeason = t.season;
  t.minute += MINUTES_PER_TICK;

  let newDay = false;
  while (t.minute >= MINUTES_PER_DAY) {
    t.minute -= MINUTES_PER_DAY;
    t.day += 1;
    newDay = true;
  }

  if (newDay) {
    t.season = seasonForDay(t.day);
    t.season_day = seasonDay(t.day);
    t.year = Math.floor((t.day - 1) / DAYS_PER_YEAR) + 1;
  }

  return {
    newDay,
    newSeason: newDay && t.season !== previousSeason,
    previousSeason,
  };
}

function isNight(world) {
  const m = world.time.minute;
  return m < 6 * 60 || m >= 20 * 60;
}

function clockString(world) {
  const m = world.time.minute;
  const hh = String(Math.floor(m / 60)).padStart(2, '0');
  const mm = String(m % 60).padStart(2, '0');
  return `${hh}:${mm}`;
}

function dayProgress(world) {
  return (world.time.minute + world.tick % 1) / MINUTES_PER_DAY;
}

/** Absolute simulated minutes since episode start (useful for time budgets). */
function elapsedMinutes(world) {
  return (world.time.day - 1) * MINUTES_PER_DAY + world.time.minute - world.time.start_minute;
}

module.exports = {
  MINUTES_PER_DAY,
  seasonForDay,
  seasonDay,
  advanceClock,
  isNight,
  clockString,
  dayProgress,
  elapsedMinutes,
};
