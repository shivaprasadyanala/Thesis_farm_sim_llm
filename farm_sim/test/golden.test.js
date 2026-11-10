'use strict';

/**
 * Golden scenarios.
 *
 * These are not exact-value snapshots (the balance will keep moving); they pin
 * the *shape* of the simulation: care pays off, neglect degrades the soil and
 * plants, and a water shortage hurts the harvest.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const { runScenario, freshSim, seedPlot } = require('./helpers');

function planted(row) {
  return row.sim.world.plots.filter((plot) => plot.crop && !plot.crop.dead);
}

function meanOf(rows, pick) {
  if (!rows.length) return 0;
  return rows.reduce((sum, row) => sum + pick(row), 0) / rows.length;
}

test('scenario: a cared-for farm ripens, harvests and sells crops', () => {
  const result = runScenario({ care: true, days: 8 });
  const metrics = result.metrics;

  assert.ok(metrics.harvests >= 3, `expected >= 3 harvests, got ${metrics.harvests}`);
  assert.ok(metrics.kg_harvested >= 8, `expected >= 8 kg, got ${metrics.kg_harvested}`);
  assert.ok(metrics.earned >= 50, `expected >= $50, got ${metrics.earned}`);
  assert.ok(metrics.water_used >= 20, 'the farmer should have irrigated');
  assert.equal(result.sim.world.objective.is_complete, true);
  assert.ok(result.sim.world.metrics.score > 100);
});

test('scenario: neglect leaves the crops stressed and thirsty', () => {
  const good = runScenario({ care: true, days: 6 });
  const neglected = runScenario({ care: false, plantOnly: true, days: 6 });

  const goodRows = planted(good);
  const lazyRows = planted(neglected);
  assert.equal(lazyRows.length, 4, 'the neglect scenario still plants four plots');

  const goodHealth = meanOf(goodRows, (row) => row.crop.health);
  const lazyHealth = meanOf(lazyRows, (row) => row.crop.health);
  const goodMoisture = meanOf(goodRows, (row) => row.soil.moisture);
  const lazyMoisture = meanOf(lazyRows, (row) => row.soil.moisture);

  assert.ok(goodHealth > 0.8, `cared crops should be healthy, got ${goodHealth.toFixed(2)}`);
  assert.ok(lazyHealth < 0.75, `neglected crops should suffer, got ${lazyHealth.toFixed(2)}`);
  assert.ok(goodHealth > lazyHealth + 0.2);
  assert.ok(goodMoisture > lazyMoisture + 0.1);
  assert.equal(neglected.metrics.water_used, 0);
});

test('scenario: a water shortage lowers the harvest', () => {
  const good = runScenario({ care: true, days: 8 });
  const drought = runScenario({ care: true, days: 8, drought: true });

  assert.ok(drought.metrics.water_used < good.metrics.water_used);
  assert.ok(
    drought.metrics.kg_harvested < good.metrics.kg_harvested,
    `drought ${drought.metrics.kg_harvested} kg should be below ${good.metrics.kg_harvested} kg`,
  );
});

test('scenario: identical seeds reproduce identical episodes', () => {
  const a = runScenario({ care: true, days: 5, seed: 4 });
  const b = runScenario({ care: true, days: 5, seed: 4 });
  assert.equal(a.digest, b.digest);
  assert.equal(a.metrics.kg_harvested, b.metrics.kg_harvested);
});

test('a dispatched rover waters a dry plot on its own', () => {
  const sim = freshSim({ preset: 'hard', seed: 2 });
  const plot = seedPlot(sim, [12, 8], 'wheat', { water: 0.08 });

  const dispatched = sim.act({ type: 'dispatch', task: 'water' });
  assert.equal(dispatched.ok, true);

  sim.world.tank.liters = sim.world.tank.capacity;
  let maxMoisture = plot.soil.moisture;
  for (let i = 0; i < 10; i += 1) {
    sim.advance({ ticks: 50 });
    maxMoisture = Math.max(maxMoisture, plot.soil.moisture);
  }

  assert.ok(sim.world.robots[0].watered > 0, 'the rover should have watered something');
  assert.ok(maxMoisture > 0.3, `the rover should have refilled the plot, peak ${maxMoisture}`);
  assert.ok(sim.world.metrics.water_used > 0, 'rover water comes from the shared tank');
  assert.ok(sim.world.robots[0].battery < 1);

  const recalled = sim.act({ type: 'dispatch', task: 'recall' });
  assert.equal(recalled.ok, true);
  assert.equal(sim.world.robots[0].task, 'idle');
});

test('the easy objective completes once two plots are planted and watered', () => {
  const sim = freshSim();
  assert.equal(sim.world.objective.is_complete, false);

  sim.world.player.tile = [12, 8];
  sim.act({ type: 'till', tile: [12, 8] });
  sim.act({ type: 'plant_crop', crop: 'wheat', tile: [12, 8] });
  sim.world.player.tile = [13, 8];
  sim.act({ type: 'till', tile: [13, 8] });
  sim.act({ type: 'plant_crop', crop: 'wheat', tile: [13, 8] });
  assert.equal(sim.world.objective.is_complete, false, 'planting alone is not enough');

  sim.world.player.tile = [3, 11];
  sim.act({ type: 'collect_water' });
  sim.world.player.tile = [12, 8];
  sim.act({ type: 'water', tile: [12, 8] });
  sim.world.player.tile = [13, 8];
  sim.act({ type: 'water', tile: [13, 8] });

  sim.advance({ ticks: 1 });
  assert.equal(sim.world.objective.is_complete, true);
  assert.ok(sim.world.player.money > sim.world.presetConfig.money);
});

test('the objective fails after its deadline', () => {
  const sim = freshSim();
  sim.advance({ ticks: 120 * (sim.world.objective.deadline_day + 1) });
  assert.equal(sim.world.objective.is_failed, true);
});
