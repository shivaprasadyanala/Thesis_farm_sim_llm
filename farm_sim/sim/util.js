'use strict';

const clamp = (v, min, max) => (v < min ? min : v > max ? max : v);
const clamp01 = (v) => clamp(v, 0, 1);
const round = (v, dp = 0) => {
  const f = 10 ** dp;
  return Math.round(v * f) / f;
};
const tileKey = (x, y) => `${x},${y}`;
const sameTile = (a, b) => !!a && !!b && a.x === b.x && a.y === b.y;
const manhattan = (a, b) => Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
const isFiniteNumber = (v) => typeof v === 'number' && Number.isFinite(v);

/** Accepts {x,y}, {tile:[x,y]}, [x,y] or separate x/y and normalises to {x,y}. */
function toTile(input, maybeY) {
  if (!input && !isFiniteNumber(maybeY)) return null;
  if (Array.isArray(input) && input.length >= 2) {
    return { x: Math.trunc(Number(input[0])), y: Math.trunc(Number(input[1])) };
  }
  if (typeof input === 'object' && input !== null) {
    if (Array.isArray(input.tile)) return toTile(input.tile);
    if (isFiniteNumber(input.x) && isFiniteNumber(input.y)) {
      return { x: Math.trunc(input.x), y: Math.trunc(input.y) };
    }
    if (isFiniteNumber(input.tx) && isFiniteNumber(input.ty)) {
      return { x: Math.trunc(input.tx), y: Math.trunc(input.ty) };
    }
  }
  if (isFiniteNumber(input) && isFiniteNumber(maybeY)) {
    return { x: Math.trunc(input), y: Math.trunc(maybeY) };
  }
  return null;
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

module.exports = {
  clamp,
  clamp01,
  round,
  tileKey,
  sameTile,
  manhattan,
  toTile,
  deepClone,
  isFiniteNumber,
};
