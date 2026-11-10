'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { createWorld } = require('../sim/world');
const cropSystem = require('../sim/systems/crop');
const { TICKS_PER_DAY } = require('../sim/config');

function controlledWorld(seed = 1) {
  const world = createWorld({ seed, preset: 'easy' });
  world.time.minute = 12 * 60; // noon: predictable light
  world.weather.temp_c = 20;
  world.weather.humidity = 0.5;
  world.weather.wind = 0.2;
  world.weather.growth_light = 1;
  world.weather.rain_rate = 0;
  world.weather.frost = false;
  return world;
}

function plantedWorld(crop = 'wheat', moisture = 0.6, seed = 1) {
  const world = controlledWorld(seed);
  const plot = world.plots[0];
  plot.tilled = true;
  plot.crop = cropSystem.makeCrop(world, crop);
  plot.soil.moisture = moisture;
  return { world, plot };
}

test('new crops start as seedlings with full health', () => {
  const { plot } = plantedWorld();
  assert.equal(plot.crop.growth, 0);
  assert.equal(plot.crop.health, 1);
  assert.equal(plot.crop.ready, false);
  assert.equal(cropSystem.stageName(plot.crop), 'seedling');
});

test('stage index advances from seedling to ready', () => {
  const { plot } = plantedWorld();
  const stages = [];
  for (const growth of [0, 0.3, 0.7, 1]) {
    plot.crop.growth = growth;
    stages.push(cropSystem.stageIndex(plot.crop));
  }
  assert.deepEqual(stages, [0, 1, 2, 3]);
  assert.equal(cropSystem.stageName(plot.crop), 'ready');
});

test('a well-watered crop ripens in roughly its nominal number of days', () => {
  const { world, plot } = plantedWorld('wheat', 0.6);
  let ticks = 0;
  while (!plot.crop.ready && ticks < 3000) {
    plot.soil.moisture = 0.6;
    cropSystem.update(world, plot);
    ticks += 1;
  }
  assert.ok(plot.crop.ready, 'crop should ripen');
  const days = ticks / TICKS_PER_DAY;
  assert.ok(days > 2.5 && days < 6.5, `ripened after ${days.toFixed(2)} days`);
});

test('dry soil grows much slower than moist soil', () => {
  const dry = plantedWorld('wheat', 0.03, 2);
  const wet = plantedWorld('wheat', 0.6, 2);

  for (let i = 0; i < 240; i += 1) {
    dry.plot.soil.moisture = 0.03;
    wet.plot.soil.moisture = 0.6;
    cropSystem.update(dry.world, dry.plot);
    cropSystem.update(wet.world, wet.plot);
  }
  assert.ok(
    dry.plot.crop.growth < wet.plot.crop.growth,
    `dry ${dry.plot.crop.growth} should lag wet ${wet.plot.crop.growth}`,
  );
});

test('persistent drought kills a crop', () => {
  const { world, plot } = plantedWorld('rice', 0.01, 3);
  let ticks = 0;
  while (!plot.crop.dead && ticks < 3000) {
    plot.soil.moisture = 0;
    cropSystem.update(world, plot);
    ticks += 1;
  }
  assert.ok(plot.crop.dead, 'crop should die from drought');
  assert.equal(plot.crop.dead_reason, 'drought_stress');
  assert.equal(world.metrics.crops_died, 1);
});

test('health recovers when conditions improve', () => {
  const { world, plot } = plantedWorld('wheat', 0.6, 4);
  plot.crop.health = 0.4;
  for (let i = 0; i < 200; i += 1) {
    plot.soil.moisture = 0.6;
    cropSystem.update(world, plot);
  }
  assert.ok(plot.crop.health > 0.4, `health should recover, got ${plot.crop.health}`);
});

test('harvest clears the plot, books metrics and returns a grade', () => {
  const { world, plot } = plantedWorld('wheat', 0.6, 5);
  plot.crop.growth = 1;
  plot.crop.ready = true;

  const result = cropSystem.harvest(world, plot);
  assert.ok(result);
  assert.ok(result.kg > 0);
  assert.equal(plot.crop, null);
  assert.equal(world.metrics.harvests, 1);
  assert.ok(Math.abs(world.metrics.kg_harvested - result.kg) < 1e-9);
  assert.ok(['A', 'B', 'C', 'D'].includes(result.grade));
});

test('weeds reduce the harvest', () => {
  const clean = plantedWorld('wheat', 0.6, 6);
  const weedy = plantedWorld('wheat', 0.6, 6);
  for (const entry of [clean, weedy]) {
    entry.plot.crop.growth = 1;
    entry.plot.crop.ready = true;
    entry.plot.crop.moisture_sum = 60;
    entry.plot.crop.moisture_samples = 100;
  }
  weedy.plot.crop.weeds = 1;

  const a = cropSystem.harvest(clean.world, clean.plot);
  const b = cropSystem.harvest(weedy.world, weedy.plot);
  assert.ok(b.kg < a.kg, `weedy ${b.kg} should be below clean ${a.kg}`);
});

test('over-ripe crops lose quality', () => {
  const { world, plot } = plantedWorld('wheat', 0.6, 7);
  plot.crop.growth = 1;
  plot.crop.ready = true;
  const before = plot.crop.quality;
  for (let i = 0; i < TICKS_PER_DAY * 3; i += 1) {
    plot.soil.moisture = 0.6;
    cropSystem.update(world, plot);
  }
  assert.ok(plot.crop.quality < before, 'quality should decay when left standing');
});

test('harvest returns null for unripe or empty plots', () => {
  const { world, plot } = plantedWorld('wheat', 0.6, 8);
  assert.equal(cropSystem.harvest(world, plot), null);
  plot.crop = null;
  assert.equal(cropSystem.harvest(world, plot), null);
});

test('disease slows growth and can be cleared', () => {
  const healthy = plantedWorld('wheat', 0.6, 9);
  const sick = plantedWorld('wheat', 0.6, 9);
  sick.plot.crop.disease = { type: 'blight', severity: 0.8 };

  for (let i = 0; i < 240; i += 1) {
    for (const entry of [healthy, sick]) {
      entry.plot.soil.moisture = 0.6;
      cropSystem.update(entry.world, entry.plot);
    }
  }
  assert.ok(sick.plot.crop.growth < healthy.plot.crop.growth);
  sick.plot.crop.disease = null;
  assert.equal(sick.plot.crop.disease, null);
});
