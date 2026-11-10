'use strict';

const { clamp, round } = require('../util');
const { pushEvent } = require('./events');

const SILO_CAPACITY = 60; // kg, shared across crops
const CONTRACT_MAX = 2;

/** Seasonal price multipliers (scarcity in winter, abundance in summer). */
const SEASONAL_PRICE = {
  spring: { wheat: 1.0, rice: 1.0, corn: 1.0, tomato: 1.05 },
  summer: { wheat: 0.95, rice: 0.95, corn: 0.95, tomato: 0.85 },
  autumn: { wheat: 0.9, rice: 0.9, corn: 0.9, tomato: 1.1 },
  winter: { wheat: 1.25, rice: 1.2, corn: 1.2, tomato: 1.35 },
};

function init(world) {
  const base = {};
  const prices = {};
  for (const [type, def] of Object.entries(world.catalog)) {
    base[type] = def.sell_price;
    prices[type] = def.sell_price;
  }
  world.market = {
    base,
    prices,
    contracts: [],
    last_refresh_day: 0,
    volatility: world.presetConfig.market_volatility,
    season: world.time.season,
  };
  world.silo = { capacity_kg: SILO_CAPACITY, items: {} };
  return world.market;
}

function totalSiloKg(world) {
  return Object.values(world.silo.items).reduce((sum, item) => sum + item.kg, 0);
}

function priceFor(world, type) {
  const base = world.market.prices[type] ?? 0;
  const seasonal = (SEASONAL_PRICE[world.time.season] || SEASONAL_PRICE.spring)[type] ?? 1;
  return round(base * seasonal, 2);
}

/** Store a harvest. Returns the accepted kg (silo capacity is shared). */
function addToSilo(world, type, kg, quality) {
  const free = Math.max(0, world.silo.capacity_kg - totalSiloKg(world));
  const accepted = round(Math.min(kg, free), 2);
  if (accepted <= 0) return 0;
  const item = world.silo.items[type] || { kg: 0, quality: 0 };
  const nextKg = round(item.kg + accepted, 2);
  item.quality = nextKg > 0 ? round((item.quality * item.kg + quality * accepted) / nextKg, 3) : quality;
  item.kg = nextKg;
  world.silo.items[type] = item;
  return accepted;
}

function buySeeds(world, type, qty) {
  const def = world.catalog[type];
  const cost = round(def.seed_cost * qty, 2);
  if (cost > world.player.money) return { ok: false, reason: 'NO_MONEY', cost };
  world.player.money = round(world.player.money - cost, 2);
  world.player.seeds[type] = (world.player.seeds[type] || 0) + qty;
  world.metrics.spent = round(world.metrics.spent + cost, 2);
  return { ok: true, cost, qty, seeds: world.player.seeds[type] };
}

function sell(world, type, kg, contractId) {
  const item = world.silo.items[type];
  if (!item || item.kg <= 0) return { ok: false, reason: 'NOTHING_TO_DO' };
  const qty = round(Math.min(kg, item.kg), 2);
  if (qty <= 0) return { ok: false, reason: 'NOTHING_TO_DO' };

  let unitPrice = priceFor(world, type) * (0.7 + 0.3 * item.quality);
  let contract = null;
  if (contractId) {
    contract = world.market.contracts.find((c) => c.id === contractId && c.crop === type && c.kg > 0);
    if (contract) unitPrice = contract.price_per_kg;
  }
  const revenue = round(qty * unitPrice, 2);

  item.kg = round(item.kg - qty, 2);
  if (item.kg <= 0.001) delete world.silo.items[type];
  if (contract) contract.kg = round(contract.kg - qty, 2);

  world.player.money = round(world.player.money + revenue, 2);
  world.metrics.earned = round(world.metrics.earned + revenue, 2);
  world.metrics.kg_sold = round(world.metrics.kg_sold + qty, 2);

  pushEvent(world, 'sale', `Sold ${qty} kg of ${type} for $${revenue}.`, {
    crop: type,
    kg: qty,
    revenue,
  });

  return { ok: true, qty, unitPrice: round(unitPrice, 2), revenue, contract: contract ? contract.id : null };
}

function refreshContracts(world) {
  world.market.contracts = world.market.contracts.filter(
    (c) => c.deadline_day > world.time.day && c.kg > 0,
  );
  while (world.market.contracts.length < CONTRACT_MAX) {
    const type = world.rng.pick(Object.keys(world.catalog));
    const kg = round(world.rng.range(4, 9), 1);
    const price = round(priceFor(world, type) * world.rng.range(1.15, 1.5), 2);
    world.market.contracts.push({
      id: `c${world.time.day}-${world.market.contracts.length + 1}`,
      crop: type,
      kg,
      price_per_kg: price,
      deadline_day: world.time.day + 3,
      issued_day: world.time.day,
    });
  }
}

function dailyUpdate(world) {
  const market = world.market;
  const volatility = market.volatility || 0;

  if (volatility > 0) {
    for (const type of Object.keys(market.prices)) {
      const base = market.base[type];
      const drift = world.rng.range(-volatility, volatility);
      market.prices[type] = round(clamp(market.prices[type] * (1 + drift), base * 0.6, base * 1.6), 2);
    }
  }

  const spoilage = world.presetConfig.spoilage || 0;
  if (spoilage > 0) {
    for (const [type, item] of Object.entries(world.silo.items)) {
      const lost = round(item.kg * spoilage, 2);
      if (lost > 0) {
        item.kg = round(Math.max(0, item.kg - lost), 2);
        if (item.kg <= 0.001) delete world.silo.items[type];
        world.metrics.kg_spoiled = round(world.metrics.kg_spoiled + lost, 2);
      }
    }
    if (world.metrics.kg_spoiled > 0 && world.time.day % 2 === 0) {
      pushEvent(world, 'spoilage', `Spoilage cost ${round(world.metrics.kg_spoiled, 1)} kg so far.`, {});
    }
  }

  if (world.time.day - market.last_refresh_day >= 2) {
    market.last_refresh_day = world.time.day;
    refreshContracts(world);
  }
  market.season = world.time.season;
}

module.exports = {
  SILO_CAPACITY,
  SEASONAL_PRICE,
  init,
  totalSiloKg,
  priceFor,
  addToSilo,
  buySeeds,
  sell,
  refreshContracts,
  dailyUpdate,
};
