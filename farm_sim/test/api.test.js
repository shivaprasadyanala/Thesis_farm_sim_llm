'use strict';

/**
 * End-to-end API test: boots the real server as a child process and exercises
 * the HTTP surface the renderer and the agents use.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const path = require('node:path');

const PORT = Number(process.env.TEST_PORT || 39217);
const BASE = `http://127.0.0.1:${PORT}`;
const ROOT = path.join(__dirname, '..');

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitForHealth(timeoutMs = 20000) {
  const started = Date.now();
  let lastError = null;
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch(`${BASE}/api/v1/health`);
      if (response.ok) return response.json();
    } catch (err) {
      lastError = err;
    }
    await sleep(200);
  }
  throw new Error(`server did not start: ${lastError ? lastError.message : 'timeout'}`);
}

async function getJson(pathname) {
  const response = await fetch(BASE + pathname);
  return { status: response.status, body: await response.json() };
}

async function postJson(pathname, payload) {
  const response = await fetch(BASE + pathname, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return { status: response.status, body: await response.json() };
}

test('HTTP API and legacy protocol', async (t) => {
  const child = spawn(process.execPath, ['server.js'], {
    cwd: ROOT,
    env: { ...process.env, PORT: String(PORT), PRESET: 'easy', SEED: '5' },
    stdio: 'ignore',
  });
  t.after(() => child.kill());

  const health = await waitForHealth();
  assert.equal(health.ok, true);
  assert.equal(health.preset, 'easy');

  await t.test('serves the Phaser client and its assets', async () => {
    const page = await fetch(`${BASE}/`);
    assert.equal(page.status, 200);
    const html = await page.text();
    assert.match(html, /Farm Sim/);
    assert.match(html, /main\.js/);

    const scene = await fetch(`${BASE}/FarmScene.js`);
    assert.equal(scene.status, 200);
    assert.match(await scene.text(), /class FarmScene/);

    const asset = await fetch(`${BASE}/assets/tiles.png`);
    assert.equal(asset.status, 200);
    assert.match(asset.headers.get('content-type') || '', /image\/png/);

    const css = await fetch(`${BASE}/styles.css`);
    assert.equal(css.status, 200);
  });

  await t.test('exposes config, schema and state', async () => {
    const config = await getJson('/api/v1/config');
    assert.equal(config.status, 200);
    assert.equal(config.body.name, 'farm-sim');
    assert.ok(config.body.presets.easy);
    assert.ok(config.body.actions.includes('water'));
    // Published so a client can budget waterings up front (the Python baseline
    // prints it in the prompt instead of hard-coding litres).
    assert.equal(config.body.presets.normal.water_per_action, 4);
    assert.ok(config.body.crops.corn.yield_kg > 0);
    // Season length and the hard preset's wallet are prompt inputs too: a four-plot
    // corn plan costs 4 x $7 and the season length decides how many cycles fit
    // before the deadline.
    assert.equal(config.body.days_per_season, 4);
    assert.equal(config.body.presets.hard.starting_money, 28);

    const schema = await getJson('/api/v1/schema');
    assert.ok(Array.isArray(schema.body.tools));
    assert.ok(schema.body.tools.length >= 10);
    assert.ok(schema.body.tools.every((tool) => tool.name && tool.parameters));

    const state = await getJson('/api/v1/state');
    assert.equal(state.status, 200);
    assert.equal(state.body.plots.length, 32);
    assert.equal(state.body.world.tile_size, 32);
    assert.ok(state.body.catalog.wheat);
    assert.ok(Array.isArray(state.body.events));
  });

  await t.test('resets a deterministic episode', async () => {
    const reset = await postJson('/api/v1/reset', { seed: 7, preset: 'normal' });
    assert.equal(reset.status, 200);
    assert.equal(reset.body.state.seed, 7);
    assert.equal(reset.body.state.preset, 'normal');
    assert.equal(reset.body.state.tick, 0);
    assert.equal(reset.body.state.time.clock, '06:00');
  });

  await t.test('applies actions and returns a state snapshot', async () => {
    const moved = await postJson('/api/v1/action', {
      type: 'move',
      dir: 'down',
      include_state: true,
    });
    assert.equal(moved.body.ok, true);
    assert.equal(moved.body.action, 'move');
    assert.deepEqual(moved.body.data.tile, [8, 5]);
    assert.deepEqual(moved.body.state.player.tile, [8, 5]);

    const failed = await postJson('/api/v1/action', { type: 'water' });
    assert.equal(failed.body.ok, false);
    assert.equal(typeof failed.body.reason, 'string');
    assert.ok(failed.body.message.length > 0);

    const unknown = await postJson('/api/v1/action', { type: 'dance' });
    assert.equal(unknown.body.reason, 'UNKNOWN_ACTION');
    assert.ok(unknown.body.data.available.includes('harvest'));

    const missing = await postJson('/api/v1/action', {});
    assert.equal(missing.body.reason, 'BAD_ARGS');
  });

  await t.test('advances the clock in bulk', async () => {
    const before = await getJson('/api/v1/state');
    const advanced = await postJson('/api/v1/advance', { minutes: 120 });
    assert.equal(advanced.body.ticks, 10);
    assert.ok(
      advanced.body.state.tick - before.body.tick >= 10,
      'ten ticks should have been applied',
    );
    assert.ok(
      advanced.body.state.time.elapsed_minutes - before.body.time.elapsed_minutes >= 120,
      'at least two simulated hours should have passed',
    );
  });

  await t.test('supports the legacy game-state protocol', async () => {
    const legacy = await getJson('/get_game_state');
    assert.equal(legacy.status, 200);
    assert.deepEqual(legacy.body.grid_size, [800, 600]);
    assert.deepEqual(legacy.body.player_pos, [200, 125]);
    assert.deepEqual(legacy.body.water_tank, [75, 250]);
    assert.equal(Object.keys(legacy.body.crops).length, 32);

    const posted = await fetch(`${BASE}/post_game_state`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        player_pos: [325, 200],
        crops: { crop1: { pos: [300, 200], name: 'wheat', planted: true, needs_water: true } },
        water_available: true,
        task: 'plant and water the crops',
      }),
    });
    const body = await posted.json();
    assert.deepEqual(body.player_pos, [325, 200]);
    assert.equal(body.crops.crop1.planted, true);
    assert.equal(body.task, 'plant and water the crops');
    assert.equal(body.water_available, true);

    const task = await postJson('/set_game_task', { task: 'new task text' });
    assert.equal(task.body.task, 'new task text');
  });

  await t.test('pause freezes the simulation', async () => {
    const paused = await postJson('/api/v1/pause', { paused: true });
    assert.equal(paused.body.paused, true);

    const blocked = await postJson('/api/v1/action', { type: 'move', dir: 'up' });
    assert.equal(blocked.body.reason, 'PAUSED');

    const advanced = await postJson('/api/v1/advance', { minutes: 60 });
    assert.equal(advanced.body.ticks, 0);

    const resumed = await postJson('/api/v1/pause', { paused: false });
    assert.equal(resumed.body.paused, false);
  });

  await t.test('unknown api routes return JSON 404', async () => {
    const response = await fetch(`${BASE}/api/v1/nope`);
    assert.equal(response.status, 404);
    const body = await response.json();
    assert.equal(body.reason, 'NOT_FOUND');
  });

  await t.test('streams state over server-sent events', async () => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 6000);
    try {
      const response = await fetch(`${BASE}/api/v1/stream`, { signal: controller.signal });
      assert.equal(response.status, 200);
      assert.match(response.headers.get('content-type') || '', /text\/event-stream/);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (!buffer.includes('event: state') && buffer.length < 40000) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
      }
      reader.cancel();
      assert.match(buffer, /event: state/);
      const payload = buffer.slice(buffer.indexOf('data: ') + 6);
      const state = JSON.parse(payload.split('\n')[0]);
      assert.equal(state.plots.length, 32);
    } finally {
      clearTimeout(timeout);
      controller.abort();
    }
  });

  // Every preset accepts every objective id (`sim/world.js` passes the override
  // into `objective.init`), so a run can hold the goal fixed and vary the world.
  await t.test('lists the objective catalogue', async () => {
    const { status, body } = await getJson('/api/v1/objectives');
    assert.equal(status, 200);
    assert.deepEqual(
      body.objectives.map((entry) => entry.id),
      ['starter_easy', 'harvest_normal', 'profit_hard'],
    );
    const harvest = body.objectives.find((entry) => entry.id === 'harvest_normal');
    assert.equal(harvest.deadline_day, 10);
    assert.equal(harvest.reward, 60);
    assert.match(harvest.text, /Harvest 3 crops/);
    // The same catalogue is embedded in the config document.
    const config = await getJson('/api/v1/config');
    assert.deepEqual(config.body.objectives, body.objectives);
    assert.equal(config.body.presets.easy.objective, 'starter_easy');
  });

  await t.test('a reset can pair a preset with a different objective', async () => {
    const mixed = await postJson('/api/v1/reset', {
      preset: 'easy',
      seed: 1,
      objective: 'harvest_normal',
    });
    assert.equal(mixed.body.state.preset, 'easy');
    // The goal comes from the override ...
    assert.equal(mixed.body.state.objective.id, 'harvest_normal');
    assert.equal(mixed.body.state.objective.reward, 60);
    assert.match(mixed.body.state.objective.text, /Harvest 3 crops/);
    // ... the task text the HUD shows follows it ...
    assert.match(mixed.body.state.task, /Harvest 3 crops/);
    // ... and the easy world (money, tank, no hazards) is untouched.
    assert.equal(mixed.body.state.player.money, 120);
    assert.equal(mixed.body.state.plots.length, 32);

    // Without the override the preset's own objective comes back.
    const restored = await postJson('/api/v1/reset', { preset: 'easy', seed: 5 });
    assert.equal(restored.body.state.objective.id, 'starter_easy');
    assert.match(restored.body.state.task, /Plant 2 crops/);

    // An unknown id is a silent fallback server-side, which is why the CLI
    // validates `--objective` against `/api/v1/objectives` before it resets.
    const unknown = await postJson('/api/v1/reset', { preset: 'easy', objective: 'nope' });
    assert.equal(unknown.body.state.objective.id, 'starter_easy');
    await postJson('/api/v1/reset', { preset: 'easy', seed: 5 });
  });
});
