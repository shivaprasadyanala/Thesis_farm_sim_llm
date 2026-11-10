'use strict';

const { Simulation, CROPS } = require('../sim');
const { plotAt } = require('../sim/world');
const { nearestPlots, Caretaker } = require('../runner');

function freshSim(options = {}) {
  return new Simulation({ seed: 1, preset: 'easy', ...options });
}

function plotOf(sim, tile) {
  return plotAt(sim.world, tile[0], tile[1]);
}

/** Plant a crop directly (bypassing movement) so a test can focus on one rule. */
function seedPlot(sim, tile, crop = 'wheat', { water = null } = {}) {
  const plot = plotOf(sim, tile);
  plot.tilled = true;
  plot.crop = require('../sim/systems/crop').makeCrop(sim.world, crop);
  if (water !== null) plot.soil.moisture = water;
  return plot;
}

function advanceDays(sim, days) {
  return sim.advance({ ticks: Math.round(days * 120) });
}

function runScenario({
  preset = 'easy',
  seed = 1,
  days = 8,
  care = true,
  drought = false,
  plantOnly = false,
} = {}) {
  const sim = new Simulation({ seed, preset });
  const tiles = nearestPlots(sim.world, 4);
  if (drought) {
    sim.world.presetConfig.tank_refill = 0;
    sim.world.tank.liters = 8;
  }
  const caretaker = care ? new Caretaker(sim, tiles) : null;
  if (caretaker) caretaker.cycle();

  if (plantOnly) {
    const planter = new Caretaker(sim, tiles);
    planter.plantAll();
  }

  for (let day = 0; day < days; day += 1) {
    for (let block = 0; block < 8; block += 1) {
      sim.advance({ ticks: 15 });
      if (caretaker) caretaker.cycle();
    }
  }
  return { sim, tiles, caretaker, metrics: sim.world.metrics, digest: sim.digest() };
}

module.exports = { freshSim, plotOf, seedPlot, advanceDays, runScenario, CROPS };
