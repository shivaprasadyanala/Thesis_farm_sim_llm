'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { createWorld } = require('../sim/world');
const soil = require('../sim/systems/soil');
const { freshSim, plotOf } = require('./helpers');

test('bare soil dries out on a sunny day', () => {
  const world = createWorld({ seed: 3, preset: 'easy' });
  const plot = world.plots[0];
  plot.soil.moisture = 1;
  world.weather.rain_rate = 0;
  for (let i = 0; i < 120; i += 1) soil.update(world, plot);
  assert.ok(plot.soil.moisture < 0.9, `expected drying, got ${plot.soil.moisture}`);
  assert.ok(plot.soil.moisture > 0.5, `drying too fast: ${plot.soil.moisture}`);
});

test('rain adds moisture and never exceeds 1', () => {
  const wet = createWorld({ seed: 3, preset: 'easy' });
  const dry = createWorld({ seed: 3, preset: 'easy' });
  wet.plots[0].soil.moisture = 0.4;
  dry.plots[0].soil.moisture = 0.4;
  wet.weather.rain_rate = 0.05;
  dry.weather.rain_rate = 0;
  for (let i = 0; i < 120; i += 1) {
    soil.update(wet, wet.plots[0]);
    soil.update(dry, dry.plots[0]);
  }
  assert.ok(wet.plots[0].soil.moisture > dry.plots[0].soil.moisture);
  assert.ok(wet.plots[0].soil.moisture <= 1);
});

test('moisture stays inside [0, 1] even under extreme weather', () => {
  const world = createWorld({ seed: 9, preset: 'hard' });
  world.weather.temp_c = 55;
  world.weather.wind = 3;
  world.weather.rain_rate = 0;
  world.plots[0].soil.moisture = 0.01;
  for (let i = 0; i < 500; i += 1) soil.update(world, world.plots[0]);
  assert.ok(world.plots[0].soil.moisture >= 0);
  assert.ok(world.plots[0].soil.moisture <= 1);
});

test('irrigation adds exactly MOISTURE_PER_LITRE per litre', () => {
  const sim = freshSim();
  const plot = plotOf(sim, [12, 8]);
  plot.soil.moisture = 0.2;
  soil.irrigate(plot, 2, sim.world);
  assert.ok(Math.abs(plot.soil.moisture - (0.2 + 2 * soil.MOISTURE_PER_LITRE)) < 1e-9);
});

test('irrigation cannot push moisture above 1', () => {
  const sim = freshSim();
  const plot = plotOf(sim, [12, 8]);
  plot.soil.moisture = 0.95;
  soil.irrigate(plot, 10, sim.world);
  assert.equal(plot.soil.moisture, 1);
});

test('needsWater follows the crop optimum', () => {
  const sim = freshSim();
  const plot = plotOf(sim, [12, 8]);
  const def = sim.world.catalog.wheat;
  plot.soil.moisture = def.optimal[0] + 0.05;
  assert.equal(soil.needsWater(plot, def), false);
  plot.soil.moisture = def.optimal[0] - 0.05;
  assert.equal(soil.needsWater(plot, def), true);
});

test('moistureBand classifies dry and soggy soils', () => {
  const sim = freshSim();
  const plot = plotOf(sim, [12, 8]);
  const def = sim.world.catalog.wheat;
  plot.soil.moisture = 0.01;
  assert.equal(soil.moistureBand(plot, def), 'critical');
  plot.soil.moisture = def.optimal[0] * 0.8;
  assert.equal(soil.moistureBand(plot, def), 'dry');
  plot.soil.moisture = (def.optimal[0] + def.optimal[1]) / 2;
  assert.equal(soil.moistureBand(plot, def), 'good');
  plot.soil.moisture = 0.99;
  assert.equal(soil.moistureBand(plot, def), 'soggy');
});
