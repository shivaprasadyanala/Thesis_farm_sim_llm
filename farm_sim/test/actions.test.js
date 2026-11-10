'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { freshSim, plotOf, seedPlot } = require('./helpers');

const at = (sim, tile) => {
  sim.world.player.tile = [...tile];
};

test('move steps exactly one tile and spends energy', () => {
  const sim = freshSim();
  const energy = sim.world.player.energy;
  const result = sim.act({ type: 'move', dir: 'right' });
  assert.equal(result.ok, true);
  assert.deepEqual(sim.world.player.tile, [9, 4]);
  assert.equal(sim.world.player.energy, energy - 1);
  assert.equal(sim.world.player.facing, 'right');
});

test('move accepts a legacy 32px / 25px delta as one tile', () => {
  const sim = freshSim();
  assert.equal(sim.act({ type: 'move', dx: 32, dy: 0 }).ok, true);
  assert.deepEqual(sim.world.player.tile, [9, 4]);
  assert.equal(sim.act({ type: 'move', dx: 0, dy: 25 }).ok, true);
  assert.deepEqual(sim.world.player.tile, [9, 5]);
});

test('diagonal moves are rejected', () => {
  const sim = freshSim();
  const result = sim.act({ type: 'move', dx: 1, dy: 1 });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'BAD_ARGS');
});

test('rocks block movement', () => {
  const sim = freshSim();
  assert.equal(sim.act({ type: 'move', dir: 'right' }).ok, true); // (9,4)
  const blocked = sim.act({ type: 'move', dir: 'right' }); // rock at (10,4)
  assert.equal(blocked.ok, false);
  assert.equal(blocked.reason, 'BLOCKED');
  assert.deepEqual(sim.world.player.tile, [9, 4]);
});

test('moving outside the world is rejected', () => {
  const sim = freshSim();
  at(sim, [0, 0]);
  const result = sim.act({ type: 'move', dir: 'left' });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'OUT_OF_BOUNDS');
});

test('move_to walks a path and reports arrival', () => {
  const sim = freshSim();
  const result = sim.act({ type: 'move_to', tile: [12, 8] });
  assert.equal(result.ok, true);
  assert.equal(result.data.reached, true);
  assert.deepEqual(sim.world.player.tile, [12, 8]);
});

test('move_to refuses a blocked destination', () => {
  const sim = freshSim();
  const result = sim.act({ type: 'move_to', tile: [10, 4] });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'NO_PATH');
});

// A 0-energy walk used to answer ok:true with `steps: 0`: the call changed
// nothing, still counted as a valid action and gave the agent no hint to rest.
test('move_to without energy fails with NO_ENERGY instead of walking nowhere', () => {
  const sim = freshSim();
  sim.world.player.energy = 0.4;
  const result = sim.act({ type: 'move_to', tile: [12, 8] });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'NO_ENERGY');
  assert.equal(result.data.steps, 0);
  assert.deepEqual(sim.world.player.tile, [8, 4], 'the farmer must not move');
  assert.equal(sim.world.metrics.invalid_actions, 1);
  assert.equal(sim.world.metrics.valid_actions, 0);
});

// The objective is re-evaluated inside `applyAction`, so the response of the
// winning action has to report it: a state that lags one tick reads as "still
// open" to an agent and makes it keep working on a finished goal.
test('the action that completes the objective already reports it', () => {
  const sim = freshSim({ objective: 'harvest_normal' });
  at(sim, [12, 9]);
  sim.world.player.energy = 100;
  const plot = seedPlot(sim, [12, 8], 'wheat');
  plot.crop.growth = 1;
  plot.crop.ready = true;
  sim.world.metrics.harvests = 2; // two crops in, the third wins the objective
  const money = sim.world.player.money;

  const result = sim.act({ type: 'harvest', tile: [12, 8] });
  assert.equal(result.ok, true);
  assert.equal(sim.observe().objective.is_complete, true, 'the response must not lag a tick');
  assert.equal(sim.world.metrics.objective_completed, 1);
  assert.equal(sim.world.player.money, money + 60, 'the $60 objective reward is paid once');
  assert.ok(result.reward >= 50, 'the winning action carries the objective bonus');
});

test('till works on an adjacent empty plot', () => {
  const sim = freshSim();
  at(sim, [12, 9]);
  const result = sim.act({ type: 'till', tile: [12, 8] });
  assert.equal(result.ok, true);
  assert.equal(plotOf(sim, [12, 8]).tilled, true);
});

test('till on a non-plot tile reports NO_PLOT', () => {
  const sim = freshSim();
  at(sim, [8, 5]);
  const result = sim.act({ type: 'till', tile: [8, 6] });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'NO_PLOT');
});

test('acting on a distant plot reports TOO_FAR', () => {
  const sim = freshSim();
  const result = sim.act({ type: 'till', tile: [19, 11] });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'TOO_FAR');
  assert.deepEqual(result.data.player, { x: 8, y: 4 });
});

test('planting requires a tilled plot', () => {
  const sim = freshSim();
  at(sim, [12, 8]);
  const result = sim.act({ type: 'plant_crop', crop: 'wheat', tile: [12, 8] });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'NOT_TILLED');
});

test('planting consumes exactly one seed', () => {
  const sim = freshSim();
  at(sim, [12, 8]);
  sim.world.player.seeds.wheat = 5;
  assert.equal(sim.act({ type: 'till', tile: [12, 8] }).ok, true);
  const result = sim.act({ type: 'plant_crop', crop: 'wheat', tile: [12, 8] });
  assert.equal(result.ok, true);
  assert.equal(sim.world.player.seeds.wheat, 4);
  assert.equal(plotOf(sim, [12, 8]).crop.type, 'wheat');
});

test('planting without seeds reports NO_SEEDS', () => {
  const sim = freshSim();
  at(sim, [12, 8]);
  sim.world.player.seeds.wheat = 0;
  sim.act({ type: 'till', tile: [12, 8] });
  const result = sim.act({ type: 'plant_crop', crop: 'wheat', tile: [12, 8] });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'NO_SEEDS');
});

test('planting on an occupied plot reports OCCUPIED', () => {
  const sim = freshSim();
  at(sim, [12, 8]);
  seedPlot(sim, [12, 8], 'wheat');
  const result = sim.act({ type: 'plant_crop', crop: 'wheat', tile: [12, 8] });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'OCCUPIED');
});

test('planting an unknown crop reports BAD_ARGS', () => {
  const sim = freshSim();
  at(sim, [12, 8]);
  sim.act({ type: 'till', tile: [12, 8] });
  const result = sim.act({ type: 'plant_crop', crop: 'banana', tile: [12, 8] });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'BAD_ARGS');
});

test('watering with an empty can reports CAN_EMPTY', () => {
  const sim = freshSim();
  at(sim, [12, 8]);
  seedPlot(sim, [12, 8], 'wheat', { water: 0.3 });
  const result = sim.act({ type: 'water', tile: [12, 8] });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'CAN_EMPTY');
  assert.equal(plotOf(sim, [12, 8]).soil.moisture, 0.3);
});

test('collect_water only works next to the tank', () => {
  const sim = freshSim();
  const far = sim.act({ type: 'collect_water' });
  assert.equal(far.ok, false);
  assert.equal(far.reason, 'NOT_AT_TANK');

  at(sim, [3, 11]);
  const near = sim.act({ type: 'collect_water' });
  assert.equal(near.ok, true);
  assert.ok(sim.world.player.can.liters > 0);
});

test('harvesting an unripe crop reports NOT_READY', () => {
  const sim = freshSim();
  at(sim, [12, 8]);
  seedPlot(sim, [12, 8], 'wheat');
  const result = sim.act({ type: 'harvest', tile: [12, 8] });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'NOT_READY');
  assert.ok(result.data.days_left > 0);
});

test('harvesting a dead crop reports CROP_DEAD', () => {
  const sim = freshSim();
  at(sim, [12, 8]);
  const plot = seedPlot(sim, [12, 8], 'wheat');
  plot.crop.dead = true;
  const result = sim.act({ type: 'harvest', tile: [12, 8] });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'CROP_DEAD');
});

test('harvesting a ripe crop stores it in the silo', () => {
  const sim = freshSim();
  at(sim, [12, 8]);
  const plot = seedPlot(sim, [12, 8], 'wheat');
  plot.crop.growth = 1;
  plot.crop.ready = true;
  const result = sim.act({ type: 'harvest', tile: [12, 8] });
  assert.equal(result.ok, true);
  assert.ok(result.data.kg > 0);
  assert.ok(sim.world.silo.items.wheat.kg > 0);
  assert.equal(plot.crop, null);
});

test('out of energy blocks physical work', () => {
  const sim = freshSim();
  at(sim, [12, 8]);
  sim.world.player.energy = 0;
  const result = sim.act({ type: 'till', tile: [12, 8] });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'NO_ENERGY');
});

test('rest restores energy and reports the farmhouse bonus', () => {
  const sim = freshSim();
  sim.world.player.energy = 5;
  const anywhere = sim.act({ type: 'rest' });
  assert.equal(anywhere.ok, true);
  assert.equal(anywhere.data.at_farmhouse, false);
  assert.equal(sim.world.player.energy, 30);

  at(sim, [2, 6]);
  const home = sim.act({ type: 'rest' });
  assert.equal(home.data.at_farmhouse, true);
  assert.equal(sim.world.player.energy, sim.world.player.max_energy);
});

test('unknown and paused actions are rejected with a reason', () => {
  const sim = freshSim();
  const unknown = sim.act({ type: 'teleport' });
  assert.equal(unknown.ok, false);
  assert.equal(unknown.reason, 'UNKNOWN_ACTION');
  assert.ok(unknown.data.available.includes('water'));

  sim.world.time.paused = true;
  const paused = sim.act({ type: 'move', dir: 'down' });
  assert.equal(paused.ok, false);
  assert.equal(paused.reason, 'PAUSED');
});

test('inspect and scan return plot details', () => {
  const sim = freshSim();
  at(sim, [12, 8]);
  seedPlot(sim, [12, 8], 'wheat', { water: 0.5 });

  const inspect = sim.act({ type: 'inspect', tile: [12, 8] });
  assert.equal(inspect.ok, true);
  assert.equal(inspect.data.plot.id, 'p01');
  assert.equal(inspect.data.plot.crop.type, 'wheat');

  const scan = sim.act({ type: 'scan', radius: 2 });
  assert.equal(scan.ok, true);
  assert.ok(scan.data.plots.length >= 1);
  assert.ok(scan.data.plots.every((plot) => plot.distance <= 2));
});

test('successful actions move the score', () => {
  const sim = freshSim();
  at(sim, [12, 8]);
  const plot = seedPlot(sim, [12, 8], 'wheat');
  plot.crop.growth = 1;
  plot.crop.ready = true;
  const result = sim.act({ type: 'harvest', tile: [12, 8] });
  assert.equal(result.ok, true);
  assert.ok(result.reward > 0, 'harvest should increase the score');
  assert.equal(sim.world.action_log.at(-1).type, 'harvest');
});

test('selling requires the silo and an item', () => {
  const sim = freshSim();
  const empty = sim.act({ type: 'sell', crop: 'wheat' });
  assert.equal(empty.ok, false);
  assert.equal(empty.reason, 'NOTHING_TO_DO');

  sim.world.silo.items.wheat = { kg: 10, quality: 0.9 };
  const far = sim.act({ type: 'sell', crop: 'wheat' });
  assert.equal(far.ok, false);
  assert.equal(far.reason, 'TOO_FAR');

  at(sim, [5, 5]);
  const sold = sim.act({ type: 'sell', crop: 'wheat', kg: 4 });
  assert.equal(sold.ok, true);
  assert.ok(sold.data.revenue > 0);
  assert.equal(sim.world.silo.items.wheat.kg, 6);
});

test('buying seeds needs money', () => {
  const sim = freshSim();
  at(sim, [5, 5]);
  sim.world.player.money = 1;
  const broke = sim.act({ type: 'buy', crop: 'wheat', qty: 2 });
  assert.equal(broke.ok, false);
  assert.equal(broke.reason, 'NO_MONEY');

  sim.world.player.money = 100;
  const bought = sim.act({ type: 'buy', crop: 'wheat', qty: 2 });
  assert.equal(bought.ok, true);
  assert.equal(sim.world.player.seeds.wheat, 8);
});

test('dispatch needs a rover', () => {
  const sim = freshSim();
  const result = sim.act({ type: 'dispatch', task: 'water' });
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'NO_ROBOT');

  const hard = freshSim({ preset: 'hard' });
  const dispatched = hard.act({ type: 'dispatch', task: 'water' });
  assert.equal(dispatched.ok, true);
  assert.equal(hard.world.robots[0].task, 'water');
});
