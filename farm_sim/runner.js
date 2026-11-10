#!/usr/bin/env node
'use strict';

/**
 * Headless runner: plays an episode with a simple scripted policy and prints
 * (optionally writes) per-day metrics. Useful for balancing and for producing
 * reproducible traces next to the LLM experiments.
 *
 *   node runner.js --preset easy --seed 1 --days 8 --policy care
 *   node runner.js --preset normal --seed 3 --days 10 --policy idle --csv out.csv
 */

const fs = require('fs');
const path = require('path');
const { Simulation, CROPS } = require('./sim');
const { plotAt } = require('./sim/world');
const soil = require('./sim/systems/soil');
const { round } = require('./sim/util');

const DEFAULT_TILES = [
  [12, 8],
  [13, 8],
  [12, 9],
  [13, 9],
];

function parseArgs(argv) {
  const args = { preset: 'easy', seed: 1, days: 8, policy: 'care', plots: 4, csv: '', quiet: false };
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (key === '--preset') args.preset = value;
    else if (key === '--seed') args.seed = Number(value);
    else if (key === '--days') args.days = Number(value);
    else if (key === '--policy') args.policy = value;
    else if (key === '--plots') args.plots = Number(value);
    else if (key === '--csv') args.csv = value;
    else if (key === '--quiet') args.quiet = true;
    else continue;
    i += 1;
  }
  return args;
}

function nearestPlots(world, count) {
  const origin = { x: 12, y: 8 };
  return world.plots
    .slice()
    .sort(
      (a, b) =>
        Math.abs(a.tile[0] - origin.x) + Math.abs(a.tile[1] - origin.y) -
        (Math.abs(b.tile[0] - origin.x) + Math.abs(b.tile[1] - origin.y)),
    )
    .slice(0, count)
    .map((plot) => [plot.tile[0], plot.tile[1]]);
}

class Caretaker {
  constructor(sim, tiles) {
    this.sim = sim;
    this.tiles = tiles;
    this.log = [];
  }

  plotAt(tile) {
    return plotAt(this.sim.world, tile[0], tile[1]);
  }

  act(action) {
    const result = this.sim.act(action);
    this.log.push(result);
    return result;
  }

  ensureWater() {
    const world = this.sim.world;
    if (world.player.can.liters >= world.presetConfig.water_per_action) return true;
    this.act({ type: 'move_to', tile: world.tank.tile });
    return this.act({ type: 'collect_water' }).ok;
  }

  restIfNeeded() {
    const world = this.sim.world;
    if (world.player.energy > 25) return;
    // Walk home for a full recovery; if that is impossible, rest on the spot.
    const home = world.landmarks.farmhouse;
    const distance = Math.abs(world.player.tile[0] - home[0]) + Math.abs(world.player.tile[1] - home[1]);
    if (world.player.energy > distance + 2) {
      this.act({ type: 'move_to', tile: home });
      this.act({ type: 'rest' });
      return;
    }
    this.act({ type: 'rest' });
  }

  plantAll() {
    for (const tile of this.tiles) {
      const plot = this.plotAt(tile);
      if (!plot || plot.crop) continue;
      this.restIfNeeded();
      this.act({ type: 'move_to', tile });
      if (!plot.tilled) this.act({ type: 'till', tile });
      const type = Object.keys(CROPS).find((crop) => (this.sim.world.player.seeds[crop] || 0) > 0);
      if (type) this.act({ type: 'plant_crop', crop: type, tile });
    }
  }

  waterDry() {
    for (const tile of this.tiles) {
      const plot = this.plotAt(tile);
      if (!plot || !plot.crop || plot.crop.dead) continue;
      const def = CROPS[plot.crop.type];
      if (!soil.needsWater(plot, def)) continue;
      this.restIfNeeded();
      if (!this.ensureWater()) break;
      this.act({ type: 'move_to', tile });
      this.act({ type: 'water', tile });
    }
  }

  harvestRipe() {
    for (const plot of this.sim.world.plots) {
      if (!plot.crop || !plot.crop.ready || plot.crop.dead) continue;
      this.act({ type: 'move_to', tile: plot.tile });
      this.act({ type: 'harvest', tile: plot.tile });
    }
  }

  sellSilo() {
    const items = Object.keys(this.sim.world.silo.items || {});
    if (!items.length) return;
    this.act({ type: 'move_to', tile: this.sim.world.landmarks.silo });
    for (const crop of items) this.act({ type: 'sell', crop });
  }

  cycle() {
    this.restIfNeeded();
    this.plantAll();
    this.waterDry();
    this.harvestRipe();
    this.sellSilo();
  }
}

function snapshot(sim, day) {
  const world = sim.world;
  const planted = world.plots.filter((p) => p.crop && !p.crop.dead);
  const avgMoisture = planted.length
    ? planted.reduce((sum, p) => sum + p.soil.moisture, 0) / planted.length
    : 0;
  return {
    day,
    season: world.time.season,
    weather: world.weather.type,
    temp_c: world.weather.temp_c,
    planted: planted.length,
    ready: planted.filter((p) => p.crop.ready).length,
    dead: world.metrics.crops_died,
    avg_moisture: round(avgMoisture, 3),
    tank_l: round(world.tank.liters, 1),
    money: world.player.money,
    harvests: world.metrics.harvests,
    kg: round(world.metrics.kg_harvested, 2),
    earned: world.metrics.earned,
    score: round(world.metrics.score || 0, 2),
    objective: world.objective.is_complete ? 'done' : world.objective.is_failed ? 'failed' : 'open',
  };
}

function main() {
  const args = parseArgs(process.argv);
  const sim = new Simulation({ seed: args.seed, preset: args.preset });
  const tiles = nearestPlots(sim.world, Math.max(1, Math.min(32, args.plots)));

  if (!args.quiet) {
    console.log(
      `farm-sim runner · preset=${args.preset} seed=${args.seed} days=${args.days} policy=${args.policy} plots=${tiles.length}`,
    );
  }

  const caretaker = args.policy === 'idle' ? null : new Caretaker(sim, tiles);
  const rows = [snapshot(sim, 1)];

  if (caretaker) caretaker.cycle();

  for (let day = 0; day < args.days; day += 1) {
    for (let block = 0; block < 8; block += 1) {
      sim.advance({ ticks: 15 }); // 3 simulated hours per block
      if (caretaker) caretaker.cycle();
    }
    rows.push(snapshot(sim, sim.world.time.day));
  }

  if (!args.quiet) {
    const header = [
      'day', 'season', 'weather', 'temp', 'planted', 'ready', 'dead',
      'moist', 'tank', 'money', 'harv', 'kg', 'earned', 'score', 'objective',
    ];
    console.log(header.join('\t'));
    for (const row of rows) {
      console.log(
        [
          row.day, row.season, row.weather, row.temp_c, row.planted, row.ready,
          row.dead, row.avg_moisture, row.tank_l, row.money, row.harvests,
          row.kg, row.earned, row.score, row.objective,
        ].join('\t'),
      );
    }
    const metrics = sim.world.metrics;
    console.log(
      `\nactions=${metrics.actions} valid=${metrics.valid_actions} invalid=${metrics.invalid_actions} ` +
        `water_used=${metrics.water_used}L harvests=${metrics.harvests} (${metrics.kg_harvested} kg) ` +
        `earned=$${metrics.earned} score=${metrics.score}`,
    );
  }

  const summary = {
    preset: args.preset,
    seed: args.seed,
    days: args.days,
    policy: args.policy,
    metrics: sim.world.metrics,
    objective: sim.world.objective,
    digest: sim.digest(),
    rows,
  };

  if (args.csv) {
    const columns = Object.keys(rows[0]);
    const csv = [
      columns.join(','),
      ...rows.map((row) => columns.map((column) => row[column]).join(',')),
    ].join('\n');
    const target = path.resolve(args.csv);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, `${csv}\n`, 'utf8');
    if (!args.quiet) console.log(`wrote ${target}`);
  }

  if (args.quiet) console.log(JSON.stringify(summary));
  return summary;
}

if (require.main === module) {
  try {
    main();
  } catch (err) {
    console.error(err);
    process.exit(1);
  }
}

module.exports = { main, Caretaker, snapshot, nearestPlots, DEFAULT_TILES };
