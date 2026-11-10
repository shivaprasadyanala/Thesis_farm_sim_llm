'use strict';

/**
 * Seeded, deterministic PRNG (mulberry32).
 *
 * Every stochastic system must draw from the world RNG so that
 * `(seed, action sequence)` fully determines an episode.
 */

function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

class Rng {
  constructor(seed = 1) {
    this.seed = normalizeSeed(seed);
    this.draws = 0;
    this._next = mulberry32(this.seed);
  }

  next() {
    this.draws += 1;
    return this._next();
  }

  /** Uniform float in [min, max). */
  range(min, max) {
    return min + (max - min) * this.next();
  }

  /** Uniform integer in [min, max] (inclusive). */
  int(min, max) {
    return Math.floor(this.range(min, max + 1));
  }

  chance(p) {
    return this.next() < p;
  }

  pick(arr) {
    if (!arr.length) return undefined;
    return arr[this.int(0, arr.length - 1)];
  }

  /** Pick a key from `{key: weight}` using the given weights. */
  weighted(weights) {
    const entries = Object.entries(weights);
    let total = 0;
    for (const [, w] of entries) total += w;
    let roll = this.next() * total;
    for (const [key, w] of entries) {
      roll -= w;
      if (roll <= 0) return key;
    }
    return entries[entries.length - 1][0];
  }

  /** Deterministic in-place shuffle (Fisher-Yates). */
  shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i -= 1) {
      const j = this.int(0, i);
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  }
}

function normalizeSeed(seed) {
  if (typeof seed === 'number' && Number.isFinite(seed)) return Math.abs(Math.floor(seed)) || 1;
  if (typeof seed === 'string' && seed.length) {
    let h = 2166136261;
    for (let i = 0; i < seed.length; i += 1) {
      h ^= seed.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return (h >>> 0) || 1;
  }
  return 1;
}

/** Cheap FNV-1a hash, used to prove two episodes are bit-identical. */
function hashString(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

module.exports = { Rng, mulberry32, normalizeSeed, hashString };
