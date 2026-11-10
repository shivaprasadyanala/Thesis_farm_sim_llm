'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { Rng, normalizeSeed, hashString } = require('../sim/rng');

test('same seed produces the same stream', () => {
  const a = new Rng(1234);
  const b = new Rng(1234);
  const left = Array.from({ length: 50 }, () => a.next());
  const right = Array.from({ length: 50 }, () => b.next());
  assert.deepEqual(left, right);
  assert.equal(a.draws, 50);
});

test('different seeds diverge', () => {
  const a = new Rng(1);
  const b = new Rng(2);
  assert.notEqual(a.next(), b.next());
});

test('range and int stay inside their bounds', () => {
  const rng = new Rng(7);
  for (let i = 0; i < 500; i += 1) {
    const value = rng.range(-2.5, 3.5);
    assert.ok(value >= -2.5 && value < 3.5, `range out of bounds: ${value}`);
    const integer = rng.int(3, 6);
    assert.ok(Number.isInteger(integer) && integer >= 3 && integer <= 6, `int out of bounds: ${integer}`);
  }
});

test('weighted respects zero weight and known outcomes', () => {
  const rng = new Rng(99);
  for (let i = 0; i < 100; i += 1) {
    assert.equal(rng.weighted({ never: 0, always: 1 }), 'always');
  }
});

test('weighted roughly follows the weights', () => {
  const rng = new Rng(2024);
  const counts = { a: 0, b: 0 };
  for (let i = 0; i < 5000; i += 1) {
    counts[rng.weighted({ a: 3, b: 1 })] += 1;
  }
  const share = counts.a / (counts.a + counts.b);
  assert.ok(share > 0.7 && share < 0.8, `unexpected share ${share}`);
});

test('chance(0) and chance(1) are constant', () => {
  const rng = new Rng(5);
  for (let i = 0; i < 100; i += 1) {
    assert.equal(rng.chance(0), false);
    assert.equal(rng.chance(1), true);
  }
});

test('normalizeSeed handles numbers and strings', () => {
  assert.equal(normalizeSeed(0), 1);
  assert.equal(normalizeSeed('abc'), normalizeSeed('abc'));
  assert.notEqual(normalizeSeed('abc'), normalizeSeed('abd'));
  assert.ok(Number.isInteger(normalizeSeed('farm')));
});

test('hashString is stable and changes with input', () => {
  assert.equal(hashString('state'), hashString('state'));
  assert.notEqual(hashString('state'), hashString('state2'));
  assert.equal(hashString('state').length, 8);
});
