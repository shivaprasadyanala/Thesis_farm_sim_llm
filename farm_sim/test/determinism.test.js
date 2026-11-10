'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { Simulation } = require('../sim');
const { tickWorld } = require('../sim/loop');
const { freshSim, seedPlot } = require('./helpers');

const at = (sim, tile) => {
  sim.world.player.tile = [...tile];
};

/** A fixed script that touches movement, soil, crops, water and economy. */
function scriptedEpisode(seed = 1, preset = 'easy') {
  const sim = new Simulation({ seed, preset });
  const at2 = (tile) => {
    sim.world.player.tile = [...tile];
  };

  for (const [index, tile] of [[0, [12, 8]], [1, [13, 8]], [2, [12, 9]]]) {
    at2(tile);
    sim.act({ type: 'till', tile });
    sim.act({ type: 'plant_crop', crop: index === 1 ? 'rice' : 'wheat', tile });
  }
  at2([3, 11]);
  sim.act({ type: 'collect_water' });
  for (const tile of [[12, 8], [13, 8], [12, 9]]) {
    at2(tile);
    sim.act({ type: 'water', tile });
  }
  for (let day = 0; day < 4; day += 1) {
    sim.advance({ ticks: 120 });
    at2([3, 11]);
    sim.act({ type: 'collect_water' });
    for (const tile of [[12, 8], [13, 8], [12, 9]]) {
      at2(tile);
      sim.act({ type: 'water', tile });
    }
  }
  return sim;
}

test('identical seeds and actions produce identical states', () => {
  const a = scriptedEpisode(42);
  const b = scriptedEpisode(42);
  assert.equal(a.digest(), b.digest());
  assert.deepEqual(a.observe().plots.map((p) => p.moisture), b.observe().plots.map((p) => p.moisture));
});

test('different seeds produce different states', () => {
  const a = scriptedEpisode(1);
  const b = scriptedEpisode(2);
  assert.notEqual(a.digest(), b.digest());
});

test('observing does not mutate the world', () => {
  const sim = freshSim();
  sim.advance({ ticks: 30 });
  const before = sim.digest();
  sim.observe();
  sim.observe({ events: 0 });
  sim.legacy();
  assert.equal(sim.digest(), before);
});

test('advance(n) equals n manual ticks', () => {
  const a = freshSim({ seed: 5 });
  const b = freshSim({ seed: 5 });
  a.advance({ ticks: 90 });
  for (let i = 0; i < 90; i += 1) tickWorld(b.world);
  assert.equal(a.digest(), b.digest());
  assert.equal(a.world.time.minute, b.world.time.minute);
});

test('advance(minutes) converts minutes into ticks', () => {
  const sim = freshSim();
  const result = sim.advance({ minutes: 360 });
  assert.equal(result.ticks, 30);
});

test('reset with the same seed reproduces the initial state', () => {
  const sim = freshSim({ seed: 11 });
  const first = sim.digest();
  sim.advance({ ticks: 300 });
  assert.notEqual(sim.digest(), first);
  sim.reset({ seed: 11, preset: 'easy' });
  assert.equal(sim.digest(), first);
});

test('weather follows the deterministic script on the easy preset', () => {
  const sim = freshSim({ seed: 3 });
  const seen = [];
  for (let day = 0; day < 4; day += 1) {
    seen.push(sim.world.weather.type);
    sim.advance({ ticks: 120 });
  }
  assert.deepEqual(seen, ['sunny', 'sunny', 'rain', 'sunny']);
});

test('hard preset hides far plots behind fog of war', () => {
  const sim = freshSim({ preset: 'hard' });
  const hidden = sim.observe().plots.filter((plot) => plot.known === false);
  assert.ok(hidden.length > 0, 'expected some plots to be unknown');
  assert.ok(hidden.every((plot) => plot.moisture === undefined));
  assert.equal(sim.observe({ fog: false }).plots.every((plot) => plot.known), true);
});

test('paused worlds do not tick', () => {
  const sim = freshSim();
  const before = sim.observe().tick;
  sim.setPaused(true);
  sim.advance({ ticks: 50 });
  assert.equal(sim.observe().tick, before);
  sim.setPaused(false);
  sim.advance({ ticks: 5 });
  assert.equal(sim.observe().tick, before + 5);
});

test('the world clock advances 12 minutes per tick and wraps at midnight', () => {
  const sim = freshSim();
  const start = sim.observe().time.minute;
  sim.advance({ ticks: 1 });
  assert.equal(sim.observe().time.minute, start + 12);
  sim.advance({ ticks: 120 });
  assert.equal(sim.observe().time.day, 2);
  assert.equal(sim.observe().time.season, 'spring');
});

test('crops are not double-counted when planted once', () => {
  const sim = freshSim();
  at(sim, [12, 8]);
  sim.act({ type: 'till', tile: [12, 8] });
  sim.act({ type: 'plant_crop', crop: 'wheat', tile: [12, 8] });
  seedPlot(sim, [12, 9], 'wheat');
  const planted = sim.observe().plots.filter((plot) => plot.crop).length;
  assert.equal(planted, 2);
});
