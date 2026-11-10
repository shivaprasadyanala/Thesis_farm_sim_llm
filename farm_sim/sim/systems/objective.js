'use strict';

const { pushEvent } = require('./events');
const { round } = require('../util');
const CROPS = require('../data/crops.json');

function countPlanted(world) {
  return world.plots.reduce((n, p) => n + (p.crop && !p.crop.dead ? 1 : 0), 0);
}

/**
 * A plot counts as watered when the player (or a rover) actually irrigated it
 * after planting and it is still moist enough - rain alone does not count.
 * `last_watered_tick` is -1 until the first irrigation (`sim/world.js`), so a
 * plot that was never watered can never look "watered after planting".
 */
function countWatered(world) {
  return world.plots.reduce((n, p) => {
    if (!p.crop || p.crop.dead) return n;
    if (p.soil.last_watered_tick < p.crop.planted_tick) return n;
    const def = CROPS[p.crop.type];
    return n + (p.soil.moisture >= def.optimal[0] * 0.9 ? 1 : 0);
  }, 0);
}

/**
 * Objectives are data + a predicate. `progress()` returns the numbers the HUD
 * and the agent see; `complete()` decides when the episode is won.
 */
const OBJECTIVES = {
  starter_easy: {
    id: 'starter_easy',
    label: 'First harvest',
    text: 'Plant 2 crops and keep them watered.',
    deadline_day: 8,
    reward: 40,
    progress(world) {
      return { planted: countPlanted(world), watered: countWatered(world) };
    },
    complete(p) {
      return p.planted >= 2 && p.watered >= 2;
    },
  },
  harvest_normal: {
    id: 'harvest_normal',
    label: 'Bring in the harvest',
    text: 'Harvest 3 crops before the end of day 10.',
    deadline_day: 10,
    reward: 60,
    progress(world) {
      return { harvests: world.metrics.harvests, planted: countPlanted(world) };
    },
    complete(p) {
      return p.harvests >= 3;
    },
  },
  profit_hard: {
    id: 'profit_hard',
    label: 'Turn a profit',
    text: 'Earn $150 from sales and harvest 4 crops before day 18.',
    deadline_day: 18,
    reward: 100,
    progress(world) {
      return {
        earnings: round(world.metrics.earned, 2),
        harvests: world.metrics.harvests,
      };
    },
    complete(p) {
      return p.earnings >= 150 && p.harvests >= 4;
    },
  },
};

function init(world, objectiveId) {
  const def = OBJECTIVES[objectiveId] || OBJECTIVES.starter_easy;
  world.objective = {
    id: def.id,
    label: def.label,
    text: def.text,
    deadline_day: def.deadline_day,
    reward: def.reward,
    progress: def.progress(world),
    is_complete: false,
    is_failed: false,
    completed_day: null,
  };
  world.objectiveDef = def;
  return world.objective;
}

function update(world) {
  const objective = world.objective;
  if (!objective || objective.is_complete || objective.is_failed) return objective;
  const def = world.objectiveDef || OBJECTIVES[objective.id];

  objective.progress = def.progress(world);

  if (!objective.is_complete && def.complete(objective.progress)) {
    objective.is_complete = true;
    objective.completed_day = world.time.day;
    world.player.money = round(world.player.money + def.reward, 2);
    world.metrics.objective_completed += 1;
    pushEvent(world, 'objective_complete', `Objective complete: ${def.label} (+$${def.reward}).`, {
      objective: def.id,
    });
    return objective;
  }

  if (!objective.is_failed && world.time.day > def.deadline_day) {
    objective.is_failed = true;
    pushEvent(world, 'objective_failed', `Objective failed: ${def.label}.`, {
      objective: def.id,
    });
  }
  return objective;
}

/**
 * Public description of every objective, for the API and the agent prompts.
 *
 * The catalogue is the single source of truth for which ids a reset may select,
 * so no client has to hard-code `starter_easy`/`harvest_normal`/`profit_hard`.
 */
function catalog() {
  return Object.values(OBJECTIVES).map((def) => ({
    id: def.id,
    label: def.label,
    text: def.text,
    deadline_day: def.deadline_day,
    reward: def.reward,
  }));
}

module.exports = { OBJECTIVES, init, update, catalog, countPlanted, countWatered };
