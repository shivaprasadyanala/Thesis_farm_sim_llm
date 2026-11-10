'use strict';

const { LANDMARKS } = require('../config');
const { findPath, blockedSet } = require('../nav');
const { irrigate } = require('./soil');
const { takeFromTank } = require('./water');
const { pushEvent } = require('./events');
const { round, clamp01 } = require('../util');
const CROPS = require('../data/crops.json');

const MOVE_EVERY_TICKS = 3;
const WATER_COST = 4; // litres per watering
const TANK_CAPACITY = 24; // litres carried by one rover
const DRAIN_IDLE = 0.0004;
const DRAIN_MOVE = 0.0022;
const CHARGE_RATE = 0.006;
const LOW_BATTERY = 0.15;
const CHARGED = 0.95;

function createRobot(world, index) {
  const tile = { x: LANDMARKS.charger.x + index, y: LANDMARKS.charger.y };
  return {
    id: `rover-${index + 1}`,
    tile: [tile.x, tile.y],
    battery: 0.9,
    task: 'idle',
    status: 'idle',
    path: [],
    cooldown: 0,
    target: null,
    resume_task: null,
    tank: { liters: 0, capacity: TANK_CAPACITY },
    watered: 0,
    distance: 0,
  };
}

function init(world) {
  world.robots = [];
  const count = world.presetConfig.robots || 0;
  for (let i = 0; i < count; i += 1) world.robots.push(createRobot(world, i));
  return world.robots;
}

function robotTile(robot) {
  return { x: robot.tile[0], y: robot.tile[1] };
}

function isAdjacent(a, b, range = 1) {
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y) <= range;
}

function pathTo(world, robot, goal) {
  const blocked = blockedSet(world);
  blocked.delete(`${goal.x},${goal.y}`);
  return findPath(robotTile(robot), goal, blocked) || [];
}

function moveAlong(world, robot) {
  robot.battery -= DRAIN_MOVE;
  if (robot.cooldown > 0) {
    robot.cooldown -= 1;
    return false;
  }
  const next = robot.path.shift();
  if (!next) return false;
  robot.tile = [next.x, next.y];
  robot.distance += 1;
  robot.cooldown = MOVE_EVERY_TICKS - 1;
  world.metrics.robot_tiles = (world.metrics.robot_tiles || 0) + 1;
  return true;
}

function plotNeedsWater(world, index) {
  const plot = world.plots[index];
  if (!plot || !plot.crop || plot.crop.dead) return false;
  const def = CROPS[plot.crop.type];
  return plot.soil.moisture < def.optimal[0] * 0.95;
}

function pickTarget(world, robot) {
  let bestIndex = null;
  let bestMoisture = Infinity;
  world.plots.forEach((plot, index) => {
    if (!plotNeedsWater(world, index)) return;
    if (plot.soil.moisture < bestMoisture) {
      bestMoisture = plot.soil.moisture;
      bestIndex = index;
    }
  });
  if (bestIndex === null) return null;
  const claimed = new Set(
    world.robots.filter((r) => r !== robot && r.target !== null).map((r) => r.target),
  );
  if (claimed.has(bestIndex)) {
    const alternative = world.plots.findIndex(
      (plot, index) => plotNeedsWater(world, index) && !claimed.has(index),
    );
    return alternative === -1 ? null : alternative;
  }
  return bestIndex;
}

function refillRobot(world, robot) {
  const room = robot.tank.capacity - robot.tank.liters;
  const litres = takeFromTank(world, room);
  robot.tank.liters = round(robot.tank.liters + litres, 2);
  robot.status = 'refilling';
  return litres;
}

function waterPlot(world, robot, plot) {
  const litres = Math.min(WATER_COST, robot.tank.liters);
  if (litres <= 0) return;
  robot.tank.liters = round(robot.tank.liters - litres, 2);
  world.metrics.water_used = round(world.metrics.water_used + litres, 2);
  irrigate(plot, litres, world);
  robot.watered += 1;
  robot.status = 'watering';
  pushEvent(world, 'robot_water', `${robot.id} watered plot ${plot.id}.`, {
    robot: robot.id,
    plot: plot.id,
  });
}

function startCharge(world, robot) {
  // Remember an interrupted watering duty so the rover can pick it up again
  // once the battery is full (an explicit `dispatch(..., 'charge')` does not).
  if (robot.task === 'water') robot.resume_task = 'water';
  robot.task = 'to_charge';
  robot.target = null;
  robot.path = pathTo(world, robot, { x: LANDMARKS.charger.x, y: LANDMARKS.charger.y });
  robot.status = 'returning';
}

function step(world, robot) {
  robot.battery = clamp01(robot.battery - DRAIN_IDLE);

  if (robot.task === 'charge' || robot.task === 'to_charge') {
    const charger = { x: LANDMARKS.charger.x, y: LANDMARKS.charger.y };
    if (isAdjacent(robotTile(robot), charger)) {
      robot.task = 'charge';
      robot.status = 'charging';
      robot.battery = clamp01(robot.battery + CHARGE_RATE);
      if (robot.battery >= CHARGED) {
        // Resume the interrupted duty instead of dropping it: a rover that goes
        // idle after charging stops watering for the rest of the episode unless
        // the agent spends another step re-dispatching it.
        if (robot.resume_task === 'water') {
          robot.task = 'water';
          robot.status = 'waiting_for_work';
          robot.target = null;
          robot.path = [];
        } else {
          robot.task = 'idle';
          robot.status = 'idle';
        }
        robot.resume_task = null;
      }
      return;
    }
    if (!robot.path.length) robot.path = pathTo(world, robot, charger);
    moveAlong(world, robot);
    robot.status = 'returning';
    return;
  }

  if (robot.battery <= LOW_BATTERY) {
    startCharge(world, robot);
    return;
  }

  if (robot.task !== 'water') {
    robot.status = 'idle';
    return;
  }

  // A rover needs water in its own tank before it can irrigate.
  if (robot.tank.liters < WATER_COST) {
    const tankTile = { x: LANDMARKS.tank.x, y: LANDMARKS.tank.y };
    if (isAdjacent(robotTile(robot), tankTile)) {
      const litres = refillRobot(world, robot);
      if (litres <= 0) {
        robot.status = 'waiting_for_water';
        robot.task = 'idle';
        pushEvent(world, 'robot_idle', `${robot.id} is out of water (tank empty).`, {
          robot: robot.id,
        });
      }
      return;
    }
    if (!robot.path.length) robot.path = pathTo(world, robot, tankTile);
    moveAlong(world, robot);
    robot.status = 'to_tank';
    return;
  }

  if (robot.target === null || !plotNeedsWater(world, robot.target)) {
    robot.target = pickTarget(world, robot);
    robot.path = [];
  }
  if (robot.target === null) {
    // Stay on duty: the rover keeps scanning for dry plots until recalled.
    robot.status = 'waiting_for_work';
    robot.path = [];
    return;
  }

  const plot = world.plots[robot.target];
  const plotTile = { x: plot.tile[0], y: plot.tile[1] };
  if (isAdjacent(robotTile(robot), plotTile)) {
    waterPlot(world, robot, plot);
    robot.target = null;
    robot.path = [];
    return;
  }
  if (!robot.path.length) robot.path = pathTo(world, robot, plotTile);
  if (!robot.path.length) {
    robot.target = null;
    robot.status = 'blocked';
    return;
  }
  moveAlong(world, robot);
  robot.status = 'to_plot';
}

function update(world) {
  for (const robot of world.robots) step(world, robot);
}

/** Uniform dispatch API used by the `dispatch` action. */
function dispatch(world, robotId, task) {
  const robot = robotId ? world.robots.find((r) => r.id === robotId) : world.robots[0];
  if (!robot) return { ok: false, reason: 'NO_ROBOT', message: 'No such rover.' };

  if (task === 'recall' || task === 'stop') {
    robot.task = 'idle';
    robot.target = null;
    robot.path = [];
    robot.status = 'idle';
    return { ok: true, robot: robot.id, task: 'idle', message: `${robot.id} recalled.` };
  }
  if (task !== 'water' && task !== 'charge') {
    return { ok: false, reason: 'BAD_ARGS', message: 'Unknown rover task.' };
  }
  if (task === 'charge') {
    startCharge(world, robot);
    return {
      ok: true,
      robot: robot.id,
      task: 'charge',
      message: `${robot.id} is returning to the charger.`,
    };
  }
  if (robot.battery <= LOW_BATTERY) {
    startCharge(world, robot);
    return {
      ok: false,
      reason: 'ROBOT_BUSY',
      robot: robot.id,
      message: `${robot.id} is low on battery and is charging first.`,
    };
  }
  robot.task = 'water';
  robot.target = null;
  robot.path = [];
  robot.status = 'dispatching';
  pushEvent(world, 'robot_dispatch', `${robot.id} dispatched to water dry plots.`, {
    robot: robot.id,
  });
  return { ok: true, robot: robot.id, task: 'water', message: `${robot.id} is heading out to water.` };
}

module.exports = {
  MOVE_EVERY_TICKS,
  WATER_COST,
  TANK_CAPACITY,
  init,
  update,
  dispatch,
  createRobot,
};
