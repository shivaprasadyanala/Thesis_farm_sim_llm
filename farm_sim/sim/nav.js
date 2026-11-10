'use strict';

const { WORLD } = require('./config');

/** Orthogonal neighbour tiles (movement is 4-directional). */
function neighbours(tile) {
  return [
    { x: tile.x + 1, y: tile.y },
    { x: tile.x - 1, y: tile.y },
    { x: tile.x, y: tile.y + 1 },
    { x: tile.x, y: tile.y - 1 },
  ];
}

function inBounds(tile) {
  return tile.x >= 0 && tile.y >= 0 && tile.x < WORLD.w && tile.y < WORLD.h;
}

/**
 * A* on the tile grid. `blocked` is a Set of "x,y" keys.
 * Returns the path *excluding* the start tile, or null when unreachable.
 */
function findPath(start, goal, blocked, maxNodes = 4000) {
  if (!start || !goal) return null;
  if (start.x === goal.x && start.y === goal.y) return [];
  const key = (t) => `${t.x},${t.y}`;
  const goalKey = key(goal);

  const open = [{ tile: start, f: 0, g: 0 }];
  const best = new Map([[key(start), 0]]);
  const cameFrom = new Map();
  let visited = 0;

  while (open.length && visited < maxNodes) {
    open.sort((a, b) => a.f - b.f);
    const current = open.shift();
    visited += 1;
    const currentKey = key(current.tile);
    if (currentKey === goalKey) {
      const path = [];
      let node = goalKey;
      while (node && node !== key(start)) {
        const [x, y] = node.split(',').map(Number);
        path.push({ x, y });
        node = cameFrom.get(node);
      }
      return path.reverse();
    }

    for (const next of neighbours(current.tile)) {
      if (!inBounds(next)) continue;
      const nextKey = key(next);
      if (blocked.has(nextKey)) continue;
      const g = current.g + 1;
      if (best.has(nextKey) && best.get(nextKey) <= g) continue;
      best.set(nextKey, g);
      cameFrom.set(nextKey, currentKey);
      const h = Math.abs(next.x - goal.x) + Math.abs(next.y - goal.y);
      open.push({ tile: next, g, f: g + h });
    }
  }
  return null;
}

/** Blocked tiles = static obstacles only (plots may be walked over). */
function blockedSet(world) {
  const set = new Set();
  for (const obstacle of world.obstacles) set.add(`${obstacle.tile[0]},${obstacle.tile[1]}`);
  return set;
}

module.exports = { neighbours, inBounds, findPath, blockedSet };
