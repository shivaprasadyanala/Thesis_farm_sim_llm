'use strict';

const { round } = require('./util');

/**
 * Single scalar used for reporting and for the per-action shaped reward.
 * Keeping it in its own module avoids a dependency cycle between the loop,
 * the action registry and the observation layer.
 */
function computeScore(world) {
  const m = world.metrics;
  return round(
    m.earned +
      8 * m.harvests +
      0.5 * m.kg_harvested +
      50 * m.objective_completed -
      0.25 * m.invalid_actions -
      2 * m.crops_died,
    2,
  );
}

module.exports = { computeScore };
