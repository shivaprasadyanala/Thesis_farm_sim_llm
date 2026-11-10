'use strict';

/**
 * HTTP surface for the farm simulation.
 *
 *  - `public/` holds the Phaser renderer (a pure view over server state)
 *  - `/api/v1/*` is the current, uniform API
 *  - `/get_game_state` + `/post_game_state` keep the original Python agents alive
 *
 * The simulation advances on a fixed 500 ms tick, independent of requests.
 */

const path = require('path');
const express = require('express');
const { Simulation, VERSION } = require('./sim');
const { MAX_TICKS_PER_ADVANCE } = require('./sim/config');

const PORT = Number(process.env.PORT) || 3111;
const HOST = process.env.HOST || '127.0.0.1';

const sim = new Simulation({
  seed: process.env.SEED === undefined ? 1 : Number(process.env.SEED),
  preset: process.env.PRESET || 'normal',
});

const app = express();
app.disable('x-powered-by');
app.use(express.json({ limit: '2mb' }));

// The agents run on the same host, but a browser on another port may also call in.
app.use('/api', (req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  if (req.method === 'OPTIONS') return res.sendStatus(204);
  return next();
});

// ---------------------------------------------------------------------------
// Access log - the 500 ms ticker stays quiet, but every call that changes the
// world is printed here, so an agent run can be followed in this terminal.
// `LOG_REQUESTS=0` silences it; GETs and the SSE stream are never logged.
// ---------------------------------------------------------------------------
const LOG_REQUESTS = process.env.LOG_REQUESTS !== '0';

app.use('/api', (req, res, next) => {
  if (!LOG_REQUESTS || req.method !== 'POST') return next();
  const started = Date.now();
  const body = req.body || {};
  const label = req.path === '/v1/reset'
    ? `reset preset=${body.preset || 'kept'} seed=${body.seed === undefined ? 'kept' : body.seed}`
    : (body.type ? `${body.type}${body.tile ? ` ${JSON.stringify(body.tile)}` : ''}` : req.originalUrl);
  const sendJson = res.json.bind(res);
  res.json = (payload) => {
    const failed = Boolean(payload) && payload.ok === false;
    const detail = [payload && payload.reason, payload && payload.message].filter(Boolean).join(' ');
    console.log(`[farm-sim] POST ${label} -> ${failed ? 'FAILED' : 'ok'}`
      + `${detail ? ` ${detail}` : ''} (${Date.now() - started} ms)`);
    return sendJson(payload);
  };
  return next();
});

app.use(express.static(path.join(__dirname, 'public')));

const wantState = (req) => {
  const flag = req.query.include_state !== undefined ? req.query.include_state : req.body && req.body.include_state;
  return flag === true || flag === '1' || flag === 'true';
};

const statePayload = (extra = {}) => ({ ...extra, state: sim.observe({ events: 6 }) });

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/api/v1/health', (req, res) => {
  res.json({
    ok: true,
    version: VERSION,
    tick: sim.world.tick,
    episode: sim.episode,
    preset: sim.world.preset,
    seed: sim.world.seed,
    uptime_s: Math.round((Date.now() - sim.createdAt) / 1000),
    clients: clients.size,
  });
});

app.get('/api/v1/config', (req, res) => {
  res.json(sim.configView());
});

// Any preset accepts any objective id, so listing them is what lets a client
// (the Python baseline, a UI) pick a goal without duplicating `objective.js`.
app.get('/api/v1/objectives', (req, res) => {
  res.json(sim.objectivesView());
});

app.get('/api/v1/schema', (req, res) => {
  res.json(sim.schemaView());
});

app.get('/api/v1/tools', (req, res) => {
  res.json({ tools: sim.schemaView().tools });
});

app.get('/api/v1/state', (req, res) => {
  const events = req.query.events === undefined ? 12 : Number(req.query.events);
  res.json(sim.observe({ events: Number.isFinite(events) ? events : 12 }));
});

app.get('/api/v1/legacy', (req, res) => {
  res.json(sim.legacy());
});

app.post('/api/v1/action', (req, res) => {
  const body = req.body || {};
  const result = sim.act(body);
  broadcast(true); // push immediately: the renderer must not wait for the next tick
  res.json(wantState(req) ? { ...result, state: sim.observe({ events: 6 }) } : result);
});

app.post('/api/v1/actions', (req, res) => {
  const list = Array.isArray(req.body) ? req.body : req.body && req.body.actions;
  if (!Array.isArray(list)) {
    return res.status(400).json({
      ok: false,
      reason: 'BAD_ARGS',
      message: 'Send an array of actions or {actions: [...]}.',
    });
  }
  const results = list.slice(0, 50).map((action) => sim.act(action));
  broadcast(true);
  return res.json({ ok: results.every((r) => r.ok), count: results.length, results });
});

app.post('/api/v1/advance', (req, res) => {
  const body = req.body || {};
  const result = sim.advance({
    ticks: body.ticks,
    minutes: body.minutes === undefined ? 60 : body.minutes,
  });
  broadcast(true);
  res.json({
    ok: true,
    ticks: result.ticks,
    days: result.days,
    clock: result.clock,
    score: result.score,
    events: result.events.map((e) => ({ type: e.type, message: e.message, day: e.day })),
    state: sim.observe({ events: 12 }),
  });
});

app.post('/api/v1/reset', (req, res) => {
  const body = req.body || {};
  const state = sim.reset({
    seed: body.seed,
    preset: body.preset,
    taskText: body.task,
    objective: body.objective,
    paused: body.paused,
    speed: body.speed,
  });
  broadcast(true);
  res.json({ ok: true, episode: sim.episode, preset: sim.world.preset, seed: sim.world.seed, state });
});

app.post('/api/v1/pause', (req, res) => {
  const paused = sim.setPaused((req.body || {}).paused !== false);
  broadcast(true);
  res.json({ ok: true, paused, state: sim.observe({ events: 4 }) });
});

app.post('/api/v1/speed', (req, res) => {
  const speed = sim.setSpeed((req.body || {}).speed);
  broadcast(true);
  res.json({ ok: true, speed });
});

app.post('/api/v1/task', (req, res) => {
  const task = sim.setTask((req.body || {}).task);
  broadcast(true);
  res.json({ ok: true, task });
});

// ---------------------------------------------------------------------------
// Server-sent events: the renderer never polls, it just applies the stream.
// ---------------------------------------------------------------------------
const clients = new Set();
let lastBroadcastTick = -1;

function sseFrame(event, payload) {
  return `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`;
}

function broadcast(force = false) {
  if (!clients.size) return;
  if (!force && sim.world.tick === lastBroadcastTick) return;
  lastBroadcastTick = sim.world.tick;
  const frame = sseFrame('state', sim.observe({ events: 8 }));
  for (const client of clients) {
    try {
      client.write(frame);
    } catch (err) {
      clients.delete(client);
    }
  }
}

app.get('/api/v1/stream', (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.write(': connected\n\n');
  res.write(sseFrame('state', sim.observe({ events: 8 })));
  clients.add(res);

  const heartbeat = setInterval(() => {
    try {
      res.write(': ping\n\n');
    } catch (err) {
      clearInterval(heartbeat);
    }
  }, 15000);

  req.on('close', () => {
    clearInterval(heartbeat);
    clients.delete(res);
  });
});

// ---------------------------------------------------------------------------
// Legacy protocol (original Phaser client + `agentic_approach*.py`)
// ---------------------------------------------------------------------------
app.get('/get_game_state', (req, res) => {
  res.json(sim.legacy());
});

app.post('/post_game_state', (req, res) => {
  const result = sim.legacyPost(req.body || {});
  broadcast(true);
  res.json(result.state);
});

app.post('/set_game_task', (req, res) => {
  const body = req.body || {};
  const task = sim.setTask(body.task || body.task_text || body.text);
  broadcast(true);
  res.json({ ok: true, task });
});

app.get('/get_game_task', (req, res) => {
  res.json({ task: sim.world.task_text, objective: sim.world.objective.text });
});

// ---------------------------------------------------------------------------
// Fixed-step scheduler
// ---------------------------------------------------------------------------
const SCHEDULER_MS = 50;
const MAX_CATCHUP_TICKS = 20;
let accumulatorMs = 0;
let lastSchedulerRun = Date.now();

setInterval(() => {
  const now = Date.now();
  const elapsed = now - lastSchedulerRun;
  lastSchedulerRun = now;

  const world = sim.world;
  if (world.time.paused) {
    accumulatorMs = 0;
    return;
  }

  const stepMs = world.tick_ms / (world.time.speed || 1);
  accumulatorMs += elapsed;

  let steps = 0;
  while (accumulatorMs >= stepMs && steps < MAX_CATCHUP_TICKS) {
    accumulatorMs -= stepMs;
    sim.tick();
    steps += 1;
  }
  // Never let a suspended process replay minutes of ticks at once.
  if (accumulatorMs > stepMs * MAX_CATCHUP_TICKS) accumulatorMs = 0;
  if (steps > 0) broadcast();
}, SCHEDULER_MS);

// ---------------------------------------------------------------------------
// Fallbacks
// ---------------------------------------------------------------------------
app.use('/api', (req, res) => {
  res.status(404).json({ ok: false, reason: 'NOT_FOUND', message: `No route for ${req.method} ${req.originalUrl}.` });
});

app.use((err, req, res, next) => {
  console.error('[farm-sim] request error:', err);
  if (res.headersSent) return next(err);
  return res.status(500).json({
    ok: false,
    reason: 'INTERNAL_ERROR',
    message: err.message || 'Something went wrong.',
  });
});

const server = app.listen(PORT, HOST, () => {
  const url = `http://${HOST}:${PORT}`;
  console.log(`[farm-sim] ${VERSION} running at ${url}`);
  console.log(`[farm-sim] preset=${sim.world.preset} seed=${sim.world.seed} tick=${sim.world.tick_ms}ms`);
  console.log('[farm-sim] state:   GET  /api/v1/state');
  console.log('[farm-sim] action:  POST /api/v1/action  { "type": "water", "tile": [12, 8] }');
  console.log('[farm-sim] advance: POST /api/v1/advance { "minutes": 720 }');
  console.log(`[farm-sim] advance is capped at ${MAX_TICKS_PER_ADVANCE} ticks per call`);
  console.log(`[farm-sim] access log ${LOG_REQUESTS ? 'on' : 'off'} (LOG_REQUESTS=0 silences it) - `
    + 'every action an agent posts is printed here');
});

function shutdown() {
  for (const client of clients) {
    try {
      client.end();
    } catch (err) {
      /* ignore */
    }
  }
  server.close(() => process.exit(0));
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

module.exports = { app, server, sim };
