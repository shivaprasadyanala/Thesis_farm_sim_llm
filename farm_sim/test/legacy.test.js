'use strict';

/**
 * The legacy `/get_game_state` + `/post_game_state` protocol must keep working
 * for the original Python agents (25px cells, `needs_water` booleans).
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const { freshSim, plotOf, seedPlot } = require('./helpers');

test('legacy view keeps the original shape on a 25px grid', () => {
  const sim = freshSim();
  const view = sim.legacy();

  assert.deepEqual(view.grid_size, [800, 600]);
  assert.deepEqual(view.player_pos, [200, 100]);
  assert.deepEqual(view.water_tank, [75, 250]);
  assert.ok(view.obstacles.some(([x, y]) => x === 250 && y === 100));
  assert.equal(Object.keys(view.crops).length, 32);
  assert.equal(view.crops.crop1.pos[0], 300);
  assert.equal(view.crops.crop1.pos[1], 200);
  assert.equal(view.water_available, false);
  assert.equal(view.goal_completed, false);
  assert.equal(typeof view.task, 'string');
});

test('legacy crop entries point at real plots', () => {
  const sim = freshSim();
  const view = sim.legacy();
  const plot = sim.world.plots[0];
  assert.deepEqual(view.crops.crop1.pos, [plot.tile[0] * 25, plot.tile[1] * 25]);
  assert.equal(view.crops.crop1.plot_id, plot.id);
});

test('legacy post can plant and water in one request', () => {
  const sim = freshSim();
  plotOf(sim, [12, 8]).soil.moisture = 0.2;
  const result = sim.legacyPost({
    player_pos: [300, 200],
    crops: {
      crop1: { pos: [300, 200], name: 'wheat', planted: true, needs_water: false },
    },
  });

  assert.deepEqual(sim.world.player.tile, [12, 8]);
  const plot = plotOf(sim, [12, 8]);
  assert.ok(plot.crop, 'crop should exist');
  assert.equal(plot.crop.type, 'wheat');
  assert.equal(plot.tilled, true, 'legacy agents never till, so we till for them');
  assert.ok(result.applied.some((entry) => entry.type === 'plant'));
  assert.ok(result.applied.some((entry) => entry.type === 'water'));
  assert.ok(plot.soil.moisture > 0.4);
  assert.equal(result.state.crops.crop1.planted, true);
  assert.equal(result.state.crops.crop1.needs_water, false);
});

test('legacy water_available flag fills the can', () => {
  const sim = freshSim();
  sim.world.tank.liters = 50;
  const result = sim.legacyPost({ water_available: true });
  assert.ok(sim.world.player.can.liters > 0);
  assert.equal(result.state.water_available, true);
});

test('legacy moves onto a rock are rejected', () => {
  const sim = freshSim();
  const result = sim.legacyPost({ player_pos: [250, 100] });
  assert.equal(result.rejected.reason, 'BLOCKED');
  assert.deepEqual(sim.world.player.tile, [8, 4]);
  assert.deepEqual(result.state.player_pos, [200, 100]);
});

test('legacy crops outside the field are rejected', () => {
  const sim = freshSim();
  const result = sim.legacyPost({
    crops: { crop9: { pos: [50, 50], name: 'wheat', planted: true } },
  });
  assert.equal(result.rejected.reason, 'NO_PLOT');
  assert.equal(sim.world.plots.filter((plot) => plot.crop).length, 0);
});

test('legacy planted=false clears a young crop', () => {
  const sim = freshSim();
  seedPlot(sim, [12, 8], 'wheat');
  const result = sim.legacyPost({
    crops: { crop1: { pos: [300, 200], planted: false } },
  });
  assert.equal(plotOf(sim, [12, 8]).crop, null);
  assert.ok(result.applied.some((entry) => entry.type === 'clear'));
});

test('legacy task text is stored and returned', () => {
  const sim = freshSim();
  const task = sim.setTask('collect water, plant the crops and water them');
  assert.equal(task, 'collect water, plant the crops and water them');
  assert.equal(sim.legacy().task, task);
  assert.equal(sim.setTask('   '), sim.world.objective.text);
});

test('legacy watering never creates water from nothing', () => {
  const sim = freshSim();
  sim.world.tank.liters = 0;
  sim.world.player.can.liters = 0;
  plotOf(sim, [12, 8]).soil.moisture = 0.05;
  const result = sim.legacyPost({
    crops: { crop1: { pos: [300, 200], name: 'wheat', planted: true, needs_water: false } },
  });
  const plot = plotOf(sim, [12, 8]);
  assert.equal(result.rejected.reason, 'CAN_EMPTY');
  assert.ok(plot.crop, 'the crop is still planted');
  assert.equal(plot.soil.last_watered_tick, -1, 'a rejected watering is not irrigation');
});
