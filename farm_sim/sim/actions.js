'use strict';

/**
 * Action registry.
 *
 * Every action validates its own preconditions and either returns
 * `{message, data}` (success) or throws `ActionError` with a machine-readable
 * reason from `ACTION_REASONS`. `applyAction` converts that into the uniform
 * envelope the HTTP API and the agents consume.
 */

const { ACTION_REASONS, LANDMARKS, TILE_SIZE, WORLD } = require('./config');
const { plotAt, isObstacle, inBounds } = require('./world');
const { toTile, manhattan, round } = require('./util');
const { findPath, blockedSet } = require('./nav');
const soilSystem = require('./systems/soil');
const cropSystem = require('./systems/crop');
const waterSystem = require('./systems/water');
const economy = require('./systems/economy');
const robots = require('./systems/robots');
const objective = require('./systems/objective');
const { pushEvent } = require('./systems/events');
const { computeScore } = require('./metrics');

const CROPS = cropSystem.CROPS;

class ActionError extends Error {
  constructor(reason, message, data) {
    super(message);
    this.name = 'ActionError';
    this.reason = reason;
    this.data = data || {};
  }
}

const ok = (action, message, data = {}) => ({ ok: true, action, reason: null, message, data });
const fail = (action, reason, message, data = {}) => ({ ok: false, action, reason, message, data });

function playerTile(world) {
  return { x: world.player.tile[0], y: world.player.tile[1] };
}

const FACING_VECTORS = {
  up: [0, -1],
  down: [0, 1],
  left: [-1, 0],
  right: [1, 0],
  north: [0, -1],
  south: [0, 1],
  west: [-1, 0],
  east: [1, 0],
};

function facingVector(dir) {
  const vector = FACING_VECTORS[String(dir || '').toLowerCase()];
  if (!vector) throw new ActionError(ACTION_REASONS.BAD_ARGS, `Unknown direction "${dir}".`);
  return vector;
}

/** The tile an action applies to when the caller does not name one. */
function defaultTarget(world) {
  const player = playerTile(world);
  if (plotAt(world, player.x, player.y)) return player;
  const [dx, dy] = facingVector(world.player.facing || 'down');
  return { x: player.x + dx, y: player.y + dy };
}

/** Accepts tile/x/y/plot id and falls back to the faced tile. */
function resolveTarget(world, args) {
  if (args.plot_id) {
    const plot = world.plots.find((p) => p.id === args.plot_id);
    if (!plot) throw new ActionError(ACTION_REASONS.NO_PLOT, `No plot with id ${args.plot_id}.`);
    return { x: plot.tile[0], y: plot.tile[1] };
  }
  const tile = toTile(args.tile) || toTile(args.x, args.y) || toTile(args.target);
  return tile || defaultTarget(world);
}

function requireBounds(tile) {
  if (!inBounds(tile.x, tile.y)) {
    throw new ActionError(ACTION_REASONS.OUT_OF_BOUNDS, `Tile ${tile.x},${tile.y} is outside the farm.`);
  }
}

function requireReach(world, tile, range = 1) {
  const player = playerTile(world);
  if (manhattan(player, tile) > range) {
    throw new ActionError(
      // A distance error, not a map edge: `OUT_OF_BOUNDS` is reserved for tiles
      // outside the farm so an agent can tell "walk closer" from "that is off-map".
      ACTION_REASONS.TOO_FAR,
      `Too far: stand within ${range} tile(s) of ${tile.x},${tile.y}.`,
      { player, target: tile },
    );
  }
}

function requirePlot(world, tile) {
  const plot = plotAt(world, tile.x, tile.y);
  if (!plot) {
    throw new ActionError(ACTION_REASONS.NO_PLOT, `Tile ${tile.x},${tile.y} is not a crop plot.`);
  }
  return plot;
}

function requireEnergy(world, amount) {
  if (world.player.energy < amount) {
    throw new ActionError(ACTION_REASONS.NO_ENERGY, 'Not enough energy - rest at the farmhouse.', {
      energy: round(world.player.energy, 1),
      required: amount,
    });
  }
}

function spendEnergy(world, amount) {
  world.player.energy = Math.max(0, world.player.energy - amount);
}

function getPlotForAction(world, args, range = 1) {
  const tile = resolveTarget(world, args);
  requireBounds(tile);
  requireReach(world, tile, range);
  return { tile, plot: requirePlot(world, tile) };
}

/** Normalise a movement argument into a single tile step. */
function stepVector(args) {
  if (args.dir !== undefined) return facingVector(args.dir);
  const rawX = Number(args.dx || 0);
  const rawY = Number(args.dy || 0);
  if (!Number.isFinite(rawX) || !Number.isFinite(rawY)) {
    throw new ActionError(ACTION_REASONS.BAD_ARGS, 'dx/dy must be numbers.');
  }
  const normalise = (value) => {
    const abs = Math.abs(value);
    if (abs === 0) return 0;
    if (abs === 1 || abs === TILE_SIZE || abs === 25) return Math.sign(value);
    return NaN;
  };
  const dx = normalise(rawX);
  const dy = normalise(rawY);
  if (Number.isNaN(dx) || Number.isNaN(dy) || Math.abs(dx) + Math.abs(dy) !== 1) {
    throw new ActionError(
      ACTION_REASONS.BAD_ARGS,
      'Move one tile at a time: pass {dir} or exactly one of dx/dy.',
      { dx: rawX, dy: rawY },
    );
  }
  return [dx, dy];
}

function movePlayerTo(world, tile) {
  world.player.tile = [tile.x, tile.y];
  world.player.moved_this_tick = true;
  world.metrics.tiles_walked = (world.metrics.tiles_walked || 0) + 1;
}

const ACTIONS = {};

function isBlocked(world, tile) {
  return isObstacle(world, tile.x, tile.y);
}

ACTIONS.move = (world, args) => {
  const [dx, dy] = stepVector(args);
  const from = playerTile(world);
  const to = { x: from.x + dx, y: from.y + dy };

  world.player.facing = dx > 0 ? 'right' : dx < 0 ? 'left' : dy > 0 ? 'down' : 'up';
  requireBounds(to);
  if (isBlocked(world, to)) {
    throw new ActionError(ACTION_REASONS.BLOCKED, `A rock blocks tile ${to.x},${to.y}.`, { target: to });
  }
  requireEnergy(world, 1);
  spendEnergy(world, 1);
  movePlayerTo(world, to);

  return ok('move', `Moved to ${to.x},${to.y}.`, {
    tile: [to.x, to.y],
    pos: [to.x * TILE_SIZE + TILE_SIZE / 2, to.y * TILE_SIZE + TILE_SIZE / 2],
    facing: world.player.facing,
  });
};

ACTIONS.move_to = (world, args) => {
  const target = toTile(args.tile) || toTile(args.x, args.y) || toTile(args.goal);
  if (!target) throw new ActionError(ACTION_REASONS.BAD_ARGS, 'move_to needs a destination tile.');
  requireBounds(target);

  const path = findPath(playerTile(world), target, blockedSet(world));
  if (!path) {
    throw new ActionError(ACTION_REASONS.NO_PATH, `No route to ${target.x},${target.y}.`, { target });
  }

  const maxSteps = Number.isFinite(Number(args.steps)) ? Math.max(1, Number(args.steps)) : path.length;
  let walked = 0;
  for (const step of path.slice(0, maxSteps)) {
    if (world.player.energy < 1) break;
    spendEnergy(world, 1);
    movePlayerTo(world, step);
    walked += 1;
  }

  const final = playerTile(world);
  const reached = final.x === target.x && final.y === target.y;
  // A walk that could not take a single step is not a success. The loop above
  // breaks on `energy < 1`, so this used to answer ok:true with `steps: 0`: it
  // counted as a valid action (`invalid_actions: 0` in the recorded `hard` run
  // while 7 of 50 calls did nothing at all) and told the model everything was
  // fine, so it retried the same call instead of resting. `move` already reports
  // NO_ENERGY for the same situation, so `move_to` must not look cheaper.
  if (walked === 0 && !reached) {
    throw new ActionError(ACTION_REASONS.NO_ENERGY, 'Out of energy - rest before walking.', {
      target,
      tile: [final.x, final.y],
      steps: 0,
      reached,
      energy: round(world.player.energy, 1),
    });
  }
  return ok(
    'move_to',
    reached ? `Arrived at ${target.x},${target.y}.` : `Walked ${walked} of ${path.length} tiles.`,
    { tile: [final.x, final.y], path_length: path.length, steps: walked, reached },
  );
};

ACTIONS.till = (world, args) => {
  const { tile, plot } = getPlotForAction(world, args);
  if (plot.crop) {
    throw new ActionError(ACTION_REASONS.OCCUPIED, `Plot ${plot.id} already has a crop.`, {
      plot: plot.id,
    });
  }
  if (plot.tilled) {
    throw new ActionError(ACTION_REASONS.ALREADY_TILLED, `Plot ${plot.id} is already tilled.`, {
      plot: plot.id,
    });
  }
  requireEnergy(world, 4);
  spendEnergy(world, 4);
  plot.tilled = true;
  pushEvent(world, 'tilled', `Tilled plot ${plot.id}.`, { plot: plot.id, tile: plot.tile });

  return ok('till', `Tilled plot ${plot.id}.`, { plot: plot.id, tile, tilled: true });
};

ACTIONS.plant_crop = (world, args) => {
  const type = String(args.seed || args.crop || args.crop_type || args.type || '').toLowerCase();
  const def = CROPS[type];
  if (!def) {
    throw new ActionError(ACTION_REASONS.BAD_ARGS, `Unknown crop type "${type}".`, {
      available: Object.keys(CROPS),
    });
  }

  const { tile, plot } = getPlotForAction(world, args);
  if (plot.crop) {
    throw new ActionError(ACTION_REASONS.OCCUPIED, `Plot ${plot.id} is already planted.`, {
      plot: plot.id,
    });
  }
  if (!plot.tilled) {
    throw new ActionError(ACTION_REASONS.NOT_TILLED, `Till plot ${plot.id} before planting.`, {
      plot: plot.id,
    });
  }
  const seeds = world.player.seeds[type] || 0;
  if (seeds <= 0) {
    throw new ActionError(ACTION_REASONS.NO_SEEDS, `No ${def.label} seeds left - buy more at the silo.`, {
      crop: type,
      seeds: 0,
    });
  }
  requireEnergy(world, def.plant_energy);
  spendEnergy(world, def.plant_energy);

  world.player.seeds[type] = seeds - 1;
  plot.crop = cropSystem.makeCrop(world, type);

  const offSeason = !def.seasons.includes(world.time.season);
  pushEvent(world, 'planted', `Planted ${def.label} on plot ${plot.id}.`, {
    plot: plot.id,
    tile: plot.tile,
    crop: type,
  });

  return ok(
    'plant_crop',
    `Planted ${def.label} on plot ${plot.id}.${offSeason ? ' (off-season: growth will be slow)' : ''}`,
    {
      plot: plot.id,
      tile,
      crop: type,
      seeds_left: world.player.seeds[type],
      off_season: offSeason,
      days_to_mature: def.days,
    },
  );
};

ACTIONS.water = (world, args) => {
  const { tile, plot } = getPlotForAction(world, args);
  const cost = world.presetConfig.water_per_action;
  const available = world.player.can.liters;
  if (available + 1e-9 < cost) {
    throw new ActionError(
      ACTION_REASONS.CAN_EMPTY,
      `Watering can is empty (${round(available, 2)} L). Collect water at the tank.`,
      { liters: round(available, 2), needed: cost },
    );
  }
  requireEnergy(world, 2);
  spendEnergy(world, 2);

  const before = plot.soil.moisture;
  waterSystem.takeFromCan(world, cost);
  soilSystem.irrigate(plot, cost, world);
  pushEvent(world, 'watered', `Watered plot ${plot.id}.`, {
    plot: plot.id,
    tile: plot.tile,
    litres: cost,
  });

  return ok('water', `Watered plot ${plot.id} with ${cost} L.`, {
    plot: plot.id,
    tile,
    litres: cost,
    moisture_before: round(before, 3),
    moisture: round(plot.soil.moisture, 3),
    can_litres: round(world.player.can.liters, 2),
  });
};

ACTIONS.collect_water = (world, args) => {
  const player = playerTile(world);
  const tankTile = { x: world.tank.tile[0], y: world.tank.tile[1] };
  if (manhattan(player, tankTile) > 1) {
    throw new ActionError(
      ACTION_REASONS.NOT_AT_TANK,
      `Stand next to the water tank at ${tankTile.x},${tankTile.y}.`,
      { player, tank: [tankTile.x, tankTile.y] },
    );
  }

  const capacity = world.player.can.capacity;
  const room = round(capacity - world.player.can.liters, 3);
  if (room <= 0.01) {
    throw new ActionError(ACTION_REASONS.NOTHING_TO_DO, 'Watering can is already full.', {
      liters: round(world.player.can.liters, 2),
      capacity,
    });
  }
  if (world.tank.liters <= 0.01) {
    throw new ActionError(ACTION_REASONS.TANK_EMPTY, 'The water tank is empty - wait for the pump.', {
      tank_liters: round(world.tank.liters, 2),
    });
  }

  const requested = Number.isFinite(Number(args.liters)) ? Number(args.liters) : room;
  const litres = waterSystem.fillCan(world, Math.min(requested, room));
  if (litres <= 0) {
    throw new ActionError(ACTION_REASONS.TANK_EMPTY, 'No water available in the tank.', {});
  }

  return ok('collect_water', `Collected ${round(litres, 2)} L of water.`, {
    litres: round(litres, 2),
    can_litres: round(world.player.can.liters, 2),
    can_capacity: capacity,
    tank_litres: round(world.tank.liters, 2),
  });
};

ACTIONS.harvest = (world, args) => {
  const { tile, plot } = getPlotForAction(world, args);
  const current = plot.crop;
  if (!current) {
    throw new ActionError(ACTION_REASONS.NO_CROP, `Plot ${plot.id} is empty.`, { plot: plot.id });
  }
  if (current.dead) {
    throw new ActionError(ACTION_REASONS.CROP_DEAD, `The crop on ${plot.id} is dead - clear it.`, {
      plot: plot.id,
    });
  }
  if (!current.ready) {
    const def = cropSystem.defFor(current.type);
    throw new ActionError(ACTION_REASONS.NOT_READY, `The ${def.label} on ${plot.id} is not ripe yet.`, {
      plot: plot.id,
      growth: round(current.growth, 3),
      stage: cropSystem.stageName(current),
      days_left: round(def.days * (1 - current.growth), 2),
    });
  }
  const def = cropSystem.defFor(current.type);
  requireEnergy(world, def.harvest_energy);

  const free = world.silo.capacity_kg - economy.totalSiloKg(world);
  if (free <= 0.05) {
    throw new ActionError(ACTION_REASONS.SILO_FULL, 'The silo is full - sell some crops first.', {
      capacity_kg: world.silo.capacity_kg,
    });
  }

  spendEnergy(world, def.harvest_energy);
  const result = cropSystem.harvest(world, plot);
  const stored = economy.addToSilo(world, result.crop, result.kg, result.quality);

  return ok(
    'harvest',
    `Harvested ${result.kg} kg of ${def.label} (grade ${result.grade}); ${stored} kg stored.`,
    { ...result, stored_kg: stored, silo_kg: round(economy.totalSiloKg(world), 2) },
  );
};

ACTIONS.clear = (world, args) => {
  const { tile, plot } = getPlotForAction(world, args);
  if (!plot.crop) {
    throw new ActionError(ACTION_REASONS.NO_CROP, `Plot ${plot.id} is already empty.`, {
      plot: plot.id,
    });
  }
  requireEnergy(world, 2);
  spendEnergy(world, 2);
  const type = plot.crop.type;
  cropSystem.clear(world, plot);
  pushEvent(world, 'cleared', `Cleared plot ${plot.id}.`, { plot: plot.id, tile: plot.tile });
  return ok('clear', `Cleared plot ${plot.id}.`, { plot: plot.id, tile, crop: type });
};

ACTIONS.weed = (world, args) => {
  const { tile, plot } = getPlotForAction(world, args);
  if (!plot.crop) {
    throw new ActionError(ACTION_REASONS.NO_CROP, `Plot ${plot.id} has no crop to weed.`, {
      plot: plot.id,
    });
  }
  if (plot.crop.weeds <= 0.01) {
    throw new ActionError(ACTION_REASONS.NOTHING_TO_DO, `Plot ${plot.id} has no weeds.`, {
      plot: plot.id,
    });
  }
  requireEnergy(world, 2);
  spendEnergy(world, 2);
  const removed = round(plot.crop.weeds, 2);
  plot.crop.weeds = 0;
  pushEvent(world, 'weeded', `Weeded plot ${plot.id}.`, { plot: plot.id, tile: plot.tile });
  return ok('weed', `Weeded plot ${plot.id}.`, { plot: plot.id, tile, removed });
};

ACTIONS.treat = (world, args) => {
  const { tile, plot } = getPlotForAction(world, args);
  if (!plot.crop || !plot.crop.disease) {
    throw new ActionError(ACTION_REASONS.NOTHING_TO_DO, `Plot ${plot.id} is not diseased.`, {
      plot: plot.id,
    });
  }
  const cost = 4;
  if (world.player.money < cost) {
    throw new ActionError(ACTION_REASONS.NO_MONEY, 'Treatment costs $4.', { cost });
  }
  requireEnergy(world, 2);
  spendEnergy(world, 2);
  world.player.money = round(world.player.money - cost, 2);
  world.metrics.spent = round(world.metrics.spent + cost, 2);
  plot.crop.disease = null;
  pushEvent(world, 'treated', `Treated plot ${plot.id}.`, { plot: plot.id, tile: plot.tile });
  return ok('treat', `Treated plot ${plot.id} for blight.`, { plot: plot.id, tile, cost });
};

ACTIONS.fertilize = (world, args) => {
  const { tile, plot } = getPlotForAction(world, args);
  const cost = 6;
  if (plot.soil.fertility >= 0.95) {
    throw new ActionError(ACTION_REASONS.NOTHING_TO_DO, `Plot ${plot.id} is already fertile.`, {
      plot: plot.id,
    });
  }
  if (world.player.money < cost) {
    throw new ActionError(ACTION_REASONS.NO_MONEY, 'Fertiliser costs $6.', { cost });
  }
  requireEnergy(world, 3);
  spendEnergy(world, 3);
  world.player.money = round(world.player.money - cost, 2);
  world.metrics.spent = round(world.metrics.spent + cost, 2);
  plot.soil.fertility = Math.min(1, round(plot.soil.fertility + 0.3, 3));
  return ok('fertilize', `Fertilised plot ${plot.id}.`, {
    plot: plot.id,
    tile,
    cost,
    fertility: plot.soil.fertility,
  });
};

function requireAtSilo(world, range = 2) {
  const silo = world.landmarks.silo;
  const player = playerTile(world);
  if (manhattan(player, { x: silo[0], y: silo[1] }) > range) {
    throw new ActionError(
      ACTION_REASONS.TOO_FAR,
      `The silo is at ${silo[0]},${silo[1]} - walk closer to trade.`,
      { player, silo },
    );
  }
}

ACTIONS.buy = (world, args) => {
  const type = String(args.seed || args.crop || args.crop_type || '').toLowerCase();
  const def = CROPS[type];
  if (!def) {
    throw new ActionError(ACTION_REASONS.BAD_ARGS, `Unknown crop type "${type}".`, {
      available: Object.keys(CROPS),
    });
  }
  const qty = Math.max(1, Math.floor(Number(args.qty || args.quantity || 1)));
  requireAtSilo(world);
  const result = economy.buySeeds(world, type, qty);
  if (!result.ok) {
    throw new ActionError(ACTION_REASONS.NO_MONEY, `Not enough money for ${qty} ${def.label} seeds.`, {
      cost: result.cost,
      money: world.player.money,
    });
  }
  return ok('buy', `Bought ${qty} ${def.label} seeds for $${result.cost}.`, {
    crop: type,
    qty,
    cost: result.cost,
    seeds: result.seeds,
    money: world.player.money,
  });
};

ACTIONS.sell = (world, args) => {
  const type = String(args.crop || args.type || args.seed || '').toLowerCase();
  const item = world.silo.items[type];
  if (!CROPS[type] || !item || item.kg <= 0) {
    throw new ActionError(ACTION_REASONS.NOTHING_TO_DO, `No ${type || 'crop'} in the silo.`, {
      silo: Object.keys(world.silo.items),
    });
  }
  requireAtSilo(world);
  const requested = Number.isFinite(Number(args.kg)) ? Number(args.kg) : item.kg;
  const result = economy.sell(world, type, requested, args.contract);
  if (!result.ok) throw new ActionError(result.reason, 'Nothing could be sold.', {});
  return ok(
    'sell',
    `Sold ${result.qty} kg of ${type} for $${result.revenue} ($${result.unitPrice}/kg).`,
    { ...result, money: world.player.money, silo_kg: round(economy.totalSiloKg(world), 2) },
  );
};

ACTIONS.dispatch = (world, args) => {
  const task = String(args.task || 'water').toLowerCase();
  const result = robots.dispatch(world, args.robot || args.robot_id, task);
  if (!result.ok) throw new ActionError(result.reason, result.message, { robot: result.robot });
  return ok('dispatch', result.message, result);
};

ACTIONS.rest = (world) => {
  const house = world.landmarks.farmhouse;
  const player = playerTile(world);
  const atHome = manhattan(player, { x: house[0], y: house[1] }) <= 1;
  if (world.player.energy >= world.player.max_energy - 0.5) {
    throw new ActionError(ACTION_REASONS.NOTHING_TO_DO, 'Already fully rested.', {
      energy: round(world.player.energy, 1),
    });
  }

  const before = world.player.energy;
  // A short break works anywhere; only the farmhouse restores you completely.
  world.player.energy = atHome
    ? world.player.max_energy
    : Math.min(world.player.max_energy, before + 25);
  world.player.resting = true;

  return ok(
    'rest',
    atHome
      ? `Rested at the farmhouse: energy ${round(before, 1)} -> ${world.player.max_energy}.`
      : `Took a short break: energy ${round(before, 1)} -> ${round(world.player.energy, 1)}.`,
    {
      energy: round(world.player.energy, 1),
      restored: round(world.player.energy - before, 1),
      at_farmhouse: atHome,
    },
  );
};

function describePlot(world, plot) {
  const def = plot.crop ? cropSystem.defFor(plot.crop.type) : null;
  return {
    id: plot.id,
    tile: plot.tile,
    tilled: plot.tilled,
    moisture: round(plot.soil.moisture, 3),
    fertility: round(plot.soil.fertility, 3),
    band: soilSystem.moistureBand(plot, def),
    needs_water: plot.crop ? soilSystem.needsWater(plot, def) : plot.soil.moisture < 0.35,
    crop: plot.crop
      ? {
          type: plot.crop.type,
          label: def.label,
          growth: round(plot.crop.growth, 3),
          stage: cropSystem.stageName(plot.crop),
          ready: plot.crop.ready,
          dead: plot.crop.dead,
          health: round(plot.crop.health, 3),
          weeds: round(plot.crop.weeds, 2),
          disease: plot.crop.disease ? plot.crop.disease.type : null,
          days_to_ripe: round(def.days * (1 - plot.crop.growth), 2),
        }
      : null,
    distance: manhattan(playerTile(world), { x: plot.tile[0], y: plot.tile[1] }),
  };
}

ACTIONS.scan = (world, args) => {
  const radius = Math.max(0, Math.min(8, Number.isFinite(Number(args.radius)) ? Number(args.radius) : 2));
  const player = playerTile(world);
  const plots = world.plots
    .filter((p) => manhattan(player, { x: p.tile[0], y: p.tile[1] }) <= radius)
    .map((p) => describePlot(world, p));
  return ok('scan', `Scanned ${plots.length} plot(s) within ${radius} tiles.`, {
    radius,
    plots,
    tank_litres: round(world.tank.liters, 2),
    can_litres: round(world.player.can.liters, 2),
  });
};

ACTIONS.inspect = (world, args) => {
  const tile = resolveTarget(world, args);
  requireBounds(tile);
  const plot = plotAt(world, tile.x, tile.y);
  if (plot) {
    return ok('inspect', `Plot ${plot.id}: ${plot.crop ? plot.crop.type : 'empty'}.`, {
      plot: describePlot(world, plot),
    });
  }
  const landmark = Object.entries(world.landmarks).find(
    ([, pos]) => pos[0] === tile.x && pos[1] === tile.y,
  );
  return ok('inspect', `Tile ${tile.x},${tile.y} is not a plot.`, {
    tile,
    obstacle: isBlocked(world, tile),
    landmark: landmark ? landmark[0] : null,
  });
};

const ACTION_NAMES = Object.keys(ACTIONS);

/**
 * Apply one action and return the uniform envelope:
 * `{ ok, action, reason, message, data, reward }`.
 */
function applyAction(world, input) {
  const payload = typeof input === 'string' ? { type: input } : input || {};
  const type = String(payload.type || payload.action || payload.name || '').trim().toLowerCase();

  if (!type) {
    return fail('(missing)', ACTION_REASONS.BAD_ARGS, 'Missing action type.', {
      available: ACTION_NAMES,
    });
  }
  const handler = ACTIONS[type];
  if (!handler) {
    return fail(type, ACTION_REASONS.UNKNOWN_ACTION, `Unknown action "${type}".`, {
      available: ACTION_NAMES,
    });
  }
  if (world.time.paused) {
    return fail(type, ACTION_REASONS.PAUSED, 'The simulation is paused.', {});
  }

  const scoreBefore = computeScore(world);
  world.metrics.actions += 1;

  let result;
  try {
    const out = handler(world, payload);
    result = { ok: true, action: type, reason: null, message: out.message, data: out.data || {} };
  } catch (err) {
    if (!(err instanceof ActionError)) throw err; // programming error, not a game rule
    result = {
      ok: false,
      action: type,
      reason: err.reason,
      message: err.message,
      data: err.data || {},
    };
  }

  if (result.ok) world.metrics.valid_actions += 1;
  else world.metrics.invalid_actions += 1;

  // Re-evaluate the goal here, not only in the tick (`loop.js`): an action can win
  // or miss it (the third harvest, the last step before the deadline) and the
  // caller must see that in the very same response. Without this the state echoed
  // with the winning `harvest` still said `is_complete: false`, so an agent that
  // had just finished was nudged with "the objective is still open" and kept
  // working. `objective.update` early-returns once complete/failed, so this cannot
  // double-pay the reward, and running it before `computeScore` makes the shaped
  // reward of the winning action carry the objective bonus instead of the next.
  objective.update(world);

  result.reward = round(computeScore(world) - scoreBefore, 2);
  world.metrics.reward = round((world.metrics.reward || 0) + result.reward, 2);

  world.action_log.push({
    tick: world.tick,
    day: world.time.day,
    minute: world.time.minute,
    type,
    ok: result.ok,
    reason: result.reason,
  });
  if (world.action_log.length > 100) world.action_log.shift();

  return result;
}

module.exports = {
  ACTIONS,
  ACTION_NAMES,
  ActionError,
  applyAction,
  describePlot,
  playerTile,
  resolveTarget,
};
