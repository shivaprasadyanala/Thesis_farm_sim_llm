/**
 * Thin client for the simulation API.
 *
 * All game state lives on the server, so this module is deliberately dumb:
 * post actions, receive state, never compute rules locally.
 */
export class FarmApi {
  constructor(base = '') {
    this.base = base;
    this.queue = Promise.resolve();
    this.source = null;
    this.pollTimer = null;
    this.onState = null;
    this.onStatus = null;
    this.failures = 0;
  }

  async json(path, options) {
    const response = await fetch(this.base + path, options);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`${response.status} ${response.statusText}: ${text.slice(0, 200)}`);
    }
    return response.json();
  }

  getConfig() {
    return this.json('/api/v1/config');
  }

  getState() {
    return this.json('/api/v1/state');
  }

  /** Serialised so two key presses cannot race each other. */
  act(action) {
    const run = () => this.json('/api/v1/action', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(action),
    });
    const result = this.queue.then(run, run);
    this.queue = result.catch(() => {});
    return result;
  }

  advance(minutes = 60) {
    return this.json('/api/v1/advance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ minutes }),
    });
  }

  reset(options = {}) {
    return this.json('/api/v1/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(options),
    });
  }

  pause(paused) {
    return this.json('/api/v1/pause', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paused }),
    });
  }

  setSpeed(speed) {
    return this.json('/api/v1/speed', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ speed }),
    });
  }

  /** Subscribe to the state stream, falling back to polling if SSE fails. */
  connect(onState, onStatus) {
    this.onState = onState;
    this.onStatus = onStatus;

    try {
      this.source = new EventSource(this.base + '/api/v1/stream');
      this.source.addEventListener('state', (event) => {
        this.failures = 0;
        onStatus('live');
        try {
          onState(JSON.parse(event.data));
        } catch (err) {
          console.error('bad state frame', err);
        }
      });
      this.source.onopen = () => onStatus('live');
      this.source.onerror = () => {
        this.failures += 1;
        onStatus('offline');
        if (this.failures >= 3) this.startPolling();
      };
    } catch (err) {
      this.startPolling();
    }
  }

  startPolling() {
    if (this.pollTimer) return;
    if (this.source) {
      this.source.close();
      this.source = null;
    }
    this.pollTimer = setInterval(async () => {
      try {
        const state = await this.getState();
        this.onStatus('live');
        this.onState(state);
      } catch (err) {
        this.onStatus('offline');
      }
    }, 1000);
  }
}
