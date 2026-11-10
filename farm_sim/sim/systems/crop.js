'use strict';

const { TICKS_PER_DAY } = require('../config');
const { clamp, clamp01, round } = require('../util');
const { isNight } = require('./time');
const { pushEvent } = require('./events');

const CROPS = require('../data/crops.json');

const STAGES = ['seedling', 'growing', 'mature', 'ready'];

/**
 * Health lost per tick at full stress.
 *
 * A crop has to survive one missed watering, otherwise the only viable plan is
 * to stand next to four plots and water every day, which no step budget can pay
 * for. At 0.002 (the old value was 0.005 - a single bone-dry day) a plot at full
 * stress takes ~4 days to die: neglect still costs yield and health, but the
 * farmer gets a day to react.
 */
const STRESS_DAMAGE = 0.002;

/** Health lost per frost tick, scaled by the crop's `frost_tolerance`. */
const FROST_DAMAGE = 0.0015;

function defFor(type) {
  return CROPS[type] || null;
}

/** Visual/gameplay stage index 0..3. */
function stageIndex(crop) {
  if (!crop) return -1;
  if (crop.dead) return -1;
  if (crop.growth >= 1) return 3;
  if (crop.growth >= 0.6) return 2;
  if (crop.growth >= 0.25) return 1;
  return 0;
}

function stageName(crop) {
  const i = stageIndex(crop);
  return i < 0 ? (crop && crop.dead ? 'dead' : 'empty') : STAGES[i];
}

function makeCrop(world, type) {
  const def = defFor(type);
  return {
    type,
    growth: 0,
    health: 1,
    quality: 1,
    weeds: 0,
    disease: null,
    dead: false,
    ready: false,
    planted_tick: world.tick,
    planted_day: world.time.day,
    ripe_ticks: 0,
    stress_ticks: 0,
    frost_damage: 0,
    moisture_sum: 0,
    moisture_samples: 0,
    yield_kg_expected: def ? def.yield_kg : 0,
  };
}

/**
 * Growth multiplier in [0, 1.2] for the current tick.
 * Water, temperature, season, light, weeds and disease all interact here.
 */
function growthFactor(world, plot) {
  const crop = plot.crop;
  if (!crop || crop.dead) return 0;
  const def = defFor(crop.type);
  if (!def) return 0;

  const moisture = plot.soil.moisture;
  const [lo, hi] = def.optimal;

  let waterF;
  if (moisture < lo) {
    // Bone dry is close to no growth at all: drought has to cost yield, not just
    // health, or skipping water is free whenever rain is around.
    waterF = (0.15 + 0.85 * clamp01(moisture / lo)) * (0.7 + 0.3 * def.drought_tolerance);
  } else if (moisture > hi) {
    waterF = clamp(1 - (moisture - hi) * 1.2, 0.4, 1);
  } else {
    waterF = 1;
  }

  const temp = world.weather.temp_c;
  const [tlo, thi] = def.temp_ideal;
  let tempF = 1;
  if (temp < tlo) tempF = clamp(1 - (tlo - temp) / 18, 0.1, 1);
  else if (temp > thi) tempF = clamp(1 - (temp - thi) / 16, 0.15, 1);
  if (!def.seasons.includes(world.time.season)) tempF *= 0.45;

  const lightF = isNight(world) ? 0.6 : world.weather.growth_light;
  const weedF = 1 - 0.35 * crop.weeds;
  const diseaseF = crop.disease ? 1 - 0.55 * crop.disease.severity : 1;
  const healthF = 0.5 + 0.5 * crop.health;

  return clamp(waterF * tempF * lightF * weedF * diseaseF * healthF, 0, 1.2);
}

/** Mean soil moisture during the crop's life, normalised to a 0..1 care score. */
function careScore(crop, def) {
  if (!crop.moisture_samples) return 0.5;
  const target = (def.optimal[0] + def.optimal[1]) / 2;
  const mean = crop.moisture_sum / crop.moisture_samples;
  const score = 1 - Math.abs(mean - target) / target;
  return clamp01(score);
}

function killCrop(world, plot, reason) {
  const crop = plot.crop;
  if (!crop || crop.dead) return false;
  crop.dead = true;
  crop.dead_reason = reason;
  world.metrics.crops_died += 1;
  pushEvent(world, 'crop_died', `${defFor(crop.type).label} died (${reason}).`, {
    plot: plot.id,
    tile: plot.tile,
    crop: crop.type,
    reason,
  });
  return true;
}

/** Per-tick crop development: growth, stress, health, ripening. */
function update(world, plot) {
  const crop = plot.crop;
  if (!crop) return null;
  const def = defFor(crop.type);
  if (!def) return null;

  if (crop.dead) {
    crop.quality = clamp01(crop.quality - 0.002);
    return null;
  }

  const moisture = plot.soil.moisture;

  if (crop.ready) {
    crop.ripe_ticks += 1;
    if (crop.ripe_ticks > TICKS_PER_DAY) crop.quality = clamp01(crop.quality - 0.0015);
    return null;
  }

  crop.moisture_sum += moisture;
  crop.moisture_samples += 1;

  const factor = growthFactor(world, plot);
  const step = (factor * (world.presetConfig.growth_speed || 1)) / (def.days * TICKS_PER_DAY);
  crop.growth = clamp01(crop.growth + step);

  // Stress accumulates health damage; recovering conditions slowly heal.
  let stress = 0;
  if (moisture < def.optimal[0] * 0.6) stress = Math.max(stress, 1 - moisture / (def.optimal[0] * 0.6));
  if (moisture > 0.98) stress = Math.max(stress, 0.4);
  if (world.weather.temp_c > def.temp_ideal[1] + 8) stress = Math.max(stress, 0.5);

  if (world.weather.frost) {
    // Per tick, so the old 0.004 deleted a nearly ripe corn in 2.5 winter days -
    // frost should hurt the harvest, not invalidate a whole planting.
    const damage = FROST_DAMAGE * (1 - def.frost_tolerance);
    crop.health -= damage;
    crop.frost_damage += damage;
    stress = Math.max(stress, 0.6);
  }

  if (stress > 0.25) {
    crop.health -= STRESS_DAMAGE * stress;
    crop.quality = clamp01(crop.quality - 0.0008 * stress);
    crop.stress_ticks += 1;
  } else {
    crop.health = clamp01(crop.health + 0.0015);
    crop.stress_ticks = 0;
  }

  crop.quality = clamp01(
    crop.quality - 0.0006 * crop.weeds - (crop.disease ? 0.0009 * crop.disease.severity : 0),
  );
  crop.health = clamp01(crop.health);

  if (crop.health <= 0.05) {
    killCrop(world, plot, crop.stress_ticks > 0 ? 'drought_stress' : 'poor_health');
    return null;
  }

  if (crop.growth >= 1 && !crop.ready) {
    crop.growth = 1;
    crop.ready = true;
    pushEvent(world, 'crop_ready', `${def.label} is ready to harvest.`, {
      plot: plot.id,
      tile: plot.tile,
      crop: crop.type,
    });
  }
  return crop;
}

/** Once-per-day hazards (weeds, disease, quality loss). */
function dailyUpdate(world, plot) {
  const crop = plot.crop;
  if (!crop || crop.dead) return;
  const def = defFor(crop.type);
  const preset = world.presetConfig;

  if (preset.weed_risk > 0) {
    const p = 0.18 * preset.weed_risk * (1 - 0.5 * crop.weeds);
    if (world.rng.chance(clamp01(p))) {
      crop.weeds = clamp01(crop.weeds + world.rng.range(0.2, 0.4));
      if (crop.weeds >= 0.6) {
        pushEvent(world, 'weeds', `Weeds are taking over a ${def.label} plot.`, {
          plot: plot.id,
          tile: plot.tile,
        });
      }
    }
  }

  if (preset.pest_risk > 0 && !crop.disease) {
    const humidityFactor = 0.25 + world.weather.humidity ** 2;
    const p = 0.12 * preset.pest_risk * def.disease_prone * humidityFactor * (1 + crop.weeds);
    if (world.rng.chance(clamp01(p))) {
      crop.disease = { type: 'blight', severity: 0.35 };
      pushEvent(world, 'disease', `${def.label} has come down with blight.`, {
        plot: plot.id,
        tile: plot.tile,
      });
    }
  } else if (crop.disease) {
    crop.disease.severity = clamp01(crop.disease.severity + 0.12);
  }

  if (crop.ready && crop.ripe_ticks > TICKS_PER_DAY) {
    crop.quality = clamp01(crop.quality - 0.02);
  }
}

/** Blight spreads to an adjacent plot with a small probability. */
function spreadDisease(world, plots) {
  if (world.presetConfig.pest_risk <= 0) return;
  const byTile = new Map(plots.map((p) => [`${p.tile[0]},${p.tile[1]}`, p]));
  for (const plot of plots) {
    const crop = plot.crop;
    if (!crop || crop.dead || !crop.disease) continue;
    const [x, y] = plot.tile;
    const neighbours = [
      [x + 1, y],
      [x - 1, y],
      [x, y + 1],
      [x, y - 1],
    ];
    for (const [nx, ny] of neighbours) {
      const other = byTile.get(`${nx},${ny}`);
      if (!other || !other.crop || other.crop.dead || other.crop.disease) continue;
      if (world.rng.chance(0.05 * crop.disease.severity)) {
        other.crop.disease = { type: 'blight', severity: 0.2 };
        pushEvent(world, 'disease', 'Blight spread to a neighbouring plot.', { plot: other.id });
      }
    }
  }
}

/** Harvest a ripe crop. Returns null when the crop cannot be harvested. */
function harvest(world, plot) {
  const crop = plot.crop;
  if (!crop || !crop.ready || crop.dead) return null;
  const def = defFor(crop.type);
  const care = careScore(crop, def);

  const kg = round(
    def.yield_kg * (0.45 + 0.55 * care) * (0.4 + 0.6 * crop.health) * (1 - 0.25 * crop.weeds),
    2,
  );
  const quality = round(clamp01(crop.quality * (0.7 + 0.3 * care)), 2);
  const grade = quality >= 0.8 ? 'A' : quality >= 0.6 ? 'B' : quality >= 0.4 ? 'C' : 'D';

  const result = {
    crop: crop.type,
    kg,
    quality,
    grade,
    care: round(care, 2),
    days: round((world.tick - crop.planted_tick) / TICKS_PER_DAY, 2),
  };

  plot.crop = null;
  plot.cleared_tick = world.tick;
  plot.soil.moisture = clamp01(plot.soil.moisture - 0.02);

  world.metrics.harvests += 1;
  world.metrics.kg_harvested = round(world.metrics.kg_harvested + kg, 2);

  pushEvent(world, 'harvest', `Harvested ${kg} kg of ${def.label} (grade ${grade}).`, {
    plot: plot.id,
    tile: plot.tile,
    crop: result.crop,
    kg,
    grade,
  });
  return result;
}

/** Remove a dead crop (or an abandoned one) so the plot can be reused. */
function clear(world, plot) {
  const crop = plot.crop;
  if (!crop) return false;
  plot.crop = null;
  plot.cleared_tick = world.tick;
  return true;
}

module.exports = {
  CROPS,
  STAGES,
  defFor,
  stageIndex,
  stageName,
  makeCrop,
  growthFactor,
  careScore,
  update,
  dailyUpdate,
  spreadDisease,
  harvest,
  clear,
  killCrop,
};
