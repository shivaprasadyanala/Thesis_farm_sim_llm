'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const water = require('../sim/systems/water');
const soil = require('../sim/systems/soil');
const { freshSim, plotOf, seedPlot } = require('./helpers');

const at = (sim, tile) => {
  sim.world.player.tile = [...tile];
};

test('collect_water moves litres from the tank into the can', () => {
  const sim = freshSim();
  const capacity = sim.world.player.can.capacity;
  at(sim, [3, 11]);
  sim.world.tank.liters = 50;
  sim.world.player.can.liters = 0;

  const result = sim.act({ type: 'collect_water' });
  assert.equal(result.ok, true);
  assert.equal(sim.world.player.can.liters, capacity);
  assert.equal(sim.world.tank.liters, 50 - capacity);
});

test('the can never exceeds its capacity', () => {
  const sim = freshSim();
  const capacity = sim.world.player.can.capacity;
  at(sim, [3, 11]);
  sim.world.player.can.liters = 8;
  sim.world.tank.liters = 100;

  const result = sim.act({ type: 'collect_water', liters: 50 });
  assert.equal(result.ok, true);
  assert.equal(sim.world.player.can.liters, capacity);
  assert.equal(sim.world.tank.liters, 100 - (capacity - 8));

  const again = sim.act({ type: 'collect_water' });
  assert.equal(again.ok, false);
  assert.equal(again.reason, 'NOTHING_TO_DO');
});

test('collecting from an empty tank reports TANK_EMPTY', () => {
  const sim = freshSim();
  at(sim, [3, 11]);
  sim.world.tank.liters = 0;
  const result = sim.act({ type: 'collect_water' });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'TANK_EMPTY');
});

test('the tank never goes negative', () => {
  const sim = freshSim();
  at(sim, [3, 11]);
  sim.world.tank.liters = 1;
  sim.world.player.can.liters = 0;
  const result = sim.act({ type: 'collect_water', liters: 10 });
  assert.equal(result.ok, true);
  assert.equal(sim.world.tank.liters, 0);
  assert.equal(sim.world.player.can.liters, 1);
});

test('watering spends exactly water_per_action and adds the matching moisture', () => {
  const sim = freshSim();
  at(sim, [12, 8]);
  const plot = seedPlot(sim, [12, 8], 'wheat', { water: 0.2 });
  sim.world.player.can.liters = 10;

  const result = sim.act({ type: 'water', tile: [12, 8] });
  assert.equal(result.ok, true);
  const cost = sim.world.presetConfig.water_per_action;
  assert.equal(sim.world.player.can.liters, 10 - cost);
  assert.ok(Math.abs(plot.soil.moisture - (0.2 + cost * soil.MOISTURE_PER_LITRE)) < 1e-6);
  assert.equal(sim.world.metrics.water_used, cost);
});

test('watering without water changes nothing', () => {
  const sim = freshSim();
  at(sim, [12, 8]);
  const plot = seedPlot(sim, [12, 8], 'wheat', { water: 0.15 });
  sim.world.player.can.liters = 1; // less than one watering

  const result = sim.act({ type: 'water', tile: [12, 8] });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'CAN_EMPTY');
  assert.equal(plot.soil.moisture, 0.15);
  assert.equal(sim.world.player.can.liters, 1);
});

test('the pump refills the tank only up to capacity', () => {
  const sim = freshSim();
  sim.world.tank.liters = 0;
  sim.world.weather.rain_rate = 0;
  const refill = sim.world.presetConfig.tank_refill;

  water.updateTank(sim.world);
  assert.ok(Math.abs(sim.world.tank.liters - refill) < 1e-9);

  sim.world.tank.liters = sim.world.tank.capacity - refill / 2;
  water.updateTank(sim.world);
  assert.equal(sim.world.tank.liters, sim.world.tank.capacity);
});

test('rain is harvested into the tank', () => {
  const sim = freshSim();
  sim.world.tank.liters = 10;
  sim.world.weather.rain_rate = 0.02;
  const before = sim.world.tank.liters;
  water.updateTank(sim.world);
  assert.ok(sim.world.tank.liters > before);
});

test('robots draw from the same tank', () => {
  const sim = freshSim({ preset: 'hard' });
  sim.world.tank.liters = 40;
  const litres = water.takeFromTank(sim.world, 8);
  assert.equal(litres, 8);
  assert.equal(sim.world.tank.liters, 32);
  assert.equal(water.takeFromTank(sim.world, 100), 32);
  assert.equal(sim.world.tank.liters, 0);
});

test('a full watering round trip conserves water', () => {
  const sim = freshSim();
  at(sim, [3, 11]);
  sim.world.tank.liters = 30;
  sim.world.player.can.liters = 0;
  const tankBefore = sim.world.tank.liters;

  sim.act({ type: 'collect_water' });
  const afterCollect = sim.world.player.can.liters;

  at(sim, [12, 8]);
  const plot = plotOf(sim, [12, 8]);
  seedPlot(sim, [12, 8], 'wheat', { water: 0.1 });
  const moistureBefore = plot.soil.moisture;
  sim.act({ type: 'water', tile: [12, 8] });

  const consumed = sim.world.player.can.capacity - sim.world.player.can.liters;
  assert.equal(consumed, sim.world.presetConfig.water_per_action);
  assert.equal(sim.world.player.can.capacity, afterCollect);
  assert.equal(
    sim.world.tank.liters + sim.world.player.can.liters,
    tankBefore - consumed,
  );
  assert.ok(
    Math.abs(plot.soil.moisture - moistureBefore - consumed * soil.MOISTURE_PER_LITRE) < 1e-6,
  );
});
