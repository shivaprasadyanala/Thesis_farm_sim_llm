'use strict';

const { clamp, clamp01, round } = require('../util');
const { pushEvent } = require('./events');

const ENERGY_PER_TILE = 1;
const REST_PER_TICK = 0.85;
const PASSIVE_REGEN = 0.06;
/** Litres caught by the tank roof per unit of rain rate. */
const RAIN_CATCH = 4;

/** Tank refill (pump + rain harvesting). Never exceeds capacity. */
function updateTank(world) {
  const tank = world.tank;
  const before = tank.liters;
  const pump = Math.min(tank.capacity - tank.liters, world.presetConfig.tank_refill);
  if (pump > 0) tank.liters += pump;
  const rain = world.weather.rain_rate * RAIN_CATCH;
  if (rain > 0) tank.liters = Math.min(tank.capacity, tank.liters + rain);
  tank.liters = Math.max(0, tank.liters);
  tank.pump_active = pump > 0;
  tank.last_gain = round(tank.liters - before, 3);
  return tank.last_gain;
}

/** Energy recovery: resting is fast, standing still is slow. */
function updateEnergy(world) {
  const player = world.player;
  if (player.resting) {
    player.energy = Math.min(player.max_energy, player.energy + REST_PER_TICK);
  } else if (!player.moved_this_tick) {
    player.energy = Math.min(player.max_energy, player.energy + PASSIVE_REGEN);
  }
  player.moved_this_tick = false;
  player.energy = Math.max(0, player.energy);
  return player.energy;
}

function spendEnergy(world, amount) {
  const player = world.player;
  if (player.energy < amount) return false;
  player.energy -= amount;
  return true;
}

/** Fill the watering can from the tank. Returns litres actually moved. */
function fillCan(world, requested) {
  const player = world.player;
  const capacity = player.can.capacity;
  const room = capacity - player.can.liters;
  const available = Math.min(room, world.tank.liters);
  const litres = Math.max(0, Math.min(requested, available));
  player.can.liters = round(player.can.liters + litres, 3);
  world.tank.liters = round(world.tank.liters - litres, 3);
  world.metrics.water_collected = round(world.metrics.water_collected + litres, 3);
  return litres;
}

/** Consume litres from the can. Returns false when there is not enough. */
function takeFromCan(world, litres) {
  const player = world.player;
  if (player.can.liters + 1e-9 < litres) return false;
  player.can.liters = round(player.can.liters - litres, 3);
  world.metrics.water_used = round(world.metrics.water_used + litres, 3);
  return true;
}

/** Draw up to `litres` from the shared tank (robots do this). */
function takeFromTank(world, litres) {
  const taken = Math.min(Math.max(0, litres), world.tank.liters);
  if (taken <= 1e-9) return 0;
  world.tank.liters = round(world.tank.liters - taken, 3);
  world.metrics.water_used = round(world.metrics.water_used + taken, 3);
  return round(taken, 3);
}

function update(world) {
  updateTank(world);
  updateEnergy(world);
}

module.exports = {
  ENERGY_PER_TILE,
  REST_PER_TICK,
  RAIN_CATCH,
  update,
  updateTank,
  updateEnergy,
  spendEnergy,
  fillCan,
  takeFromCan,
  takeFromTank,
};
