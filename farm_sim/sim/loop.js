'use strict';

const { advanceClock } = require('./systems/time');
const weather = require('./systems/weather');
const soil = require('./systems/soil');
const crop = require('./systems/crop');
const water = require('./systems/water');
const economy = require('./systems/economy');
const robots = require('./systems/robots');
const objective = require('./systems/objective');
const { beginTick } = require('./world');
const { round } = require('./util');
const { computeScore } = require('./metrics');

/**
 * Advance the world by one tick (12 simulated minutes).
 * Order matters: time -> weather -> water -> soil -> crops -> robots -> economy.
 */
function tickWorld(world) {
  if (world.time.paused) return { skipped: true, newDay: false, newSeason: false };

  world.tick += 1;
  beginTick(world);

  const clock = advanceClock(world);
  weather.update(world, clock);
  water.update(world);

  for (const plot of world.plots) {
    soil.update(world, plot);
    crop.update(world, plot);
  }

  robots.update(world);

  if (clock.newDay) {
    for (const plot of world.plots) crop.dailyUpdate(world, plot);
    crop.spreadDisease(world, world.plots);
    economy.dailyUpdate(world);
  }

  objective.update(world);
  world.metrics.score = computeScore(world);

  return { skipped: false, ...clock };
}

/**
 * Run `n` ticks, collecting the events produced on the way.
 * Used by `POST /api/v1/advance` and the headless runner.
 */
function advanceTicks(world, n) {
  const count = Math.max(0, Math.min(Math.floor(n) || 0, 5000));
  const before = world.events.length;
  const startTick = world.tick;
  const daysSeen = new Set();
  for (let i = 0; i < count; i += 1) {
    const result = tickWorld(world);
    if (result.newDay) daysSeen.add(world.time.day);
    if (world.time.paused) break;
  }
  return {
    ticks: world.tick - startTick,
    events: world.events.slice(before),
    days: [...daysSeen],
    tick: world.tick,
  };
}

module.exports = { tickWorld, advanceTicks, computeScore };
