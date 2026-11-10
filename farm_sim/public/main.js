/**
 * Boot sequence:
 *   1. read config + initial state from the server
 *   2. build the Phaser game with that state (the renderer is a pure view)
 *   3. stream state over SSE and forward intents to /api/v1/action
 */

import { FarmApi } from './api.js';
import { Hud } from './hud.js';
import { FarmScene } from './FarmScene.js';

const api = new FarmApi();
const hud = new Hud();

// Surface client crashes in the DOM so a headless smoke test can see them.
window.addEventListener('error', (event) => {
  document.body.dataset.clientError = String(event.message || 'error').slice(0, 240);
});
window.addEventListener('unhandledrejection', (event) => {
  const reason = event.reason && event.reason.message ? event.reason.message : String(event.reason);
  document.body.dataset.clientError = `unhandled: ${reason}`.slice(0, 240);
});

/** Actions that debit or credit `player.money` (harvesting only fills the silo). */
const MONEY_ACTIONS = new Set(['buy', 'sell', 'treat', 'fertilize']);

/** `Sold 2.3 kg ...` → `Sold 2.3 kg ...  ·  balance $137.55` for money actions. */
function withBalance(result, action) {
  if (!MONEY_ACTIONS.has(action.type) || !result.state || !result.state.player) {
    return result.message;
  }
  return `${result.message}  ·  balance $${result.state.player.money.toFixed(2)}`;
}

/** Turn a machine reason into a next step a player can act on. */
function describeFailure(result) {
  const state = result.state;
  const balance = state && state.player ? ` (balance $${state.player.money.toFixed(2)})` : '';
  if (result.reason === 'SILO_FULL') {
    const capacity = state && state.silo ? ` (${state.silo.capacity_kg} kg)` : '';
    return `Silo full${capacity} — walk to the barn at 5,4 and press G to sell.`;
  }
  if (result.reason === 'NO_MONEY') return `${result.message}${balance}`;
  return result.message;
}

async function boot() {
  let config;
  let state;

  try {
    config = await api.getConfig();
    state = await api.getState();
  } catch (err) {
    document.getElementById('hud-task').textContent =
      `Cannot reach the simulation server: ${err.message}. Start it with "npm start".`;
    hud.setConnection('offline');
    return;
  }

  hud.update(state, {});
  hud.setConnection('live');
  hud.toast(
    `Connected · ${config.presets[state.preset] ? config.presets[state.preset].label : state.preset} · seed ${state.seed}`,
    'ok',
  );

  const game = new Phaser.Game({
    type: Phaser.AUTO,
    parent: 'game-container',
    backgroundColor: '#2c4a2f',
    pixelArt: true,
    scale: {
      mode: Phaser.Scale.RESIZE,
      width: window.innerWidth,
      height: window.innerHeight,
    },
  });

  const scene = new FarmScene();

  const onAction = async (action) => {
    try {
      const result = await api.act({ ...action, include_state: true });
      if (result.state) scene.applyState(result.state);
      if (!result.ok) {
        hud.toast(describeFailure(result), 'err');
      } else if (action.type !== 'move') {
        hud.toast(withBalance(result, action), 'ok');
      }
      return result;
    } catch (err) {
      hud.toast(`Action failed: ${err.message}`, 'err');
      return { ok: false, message: err.message };
    }
  };

  const onPause = async () => {
    const current = Boolean(scene.state && scene.state.time.paused);
    try {
      const result = await api.pause(!current);
      if (result.state) scene.applyState(result.state);
      hud.toast(result.paused ? 'Simulation paused' : 'Simulation resumed', 'ok');
    } catch (err) {
      hud.toast(`Pause failed: ${err.message}`, 'err');
    }
  };

  const onSpeed = async (delta) => {
    const current = (scene.state && scene.state.time.speed) || 1;
    const next = Math.max(1, Math.min(8, current + delta));
    try {
      const result = await api.setSpeed(next);
      hud.toast(`Speed ${result.speed}x`, 'ok');
    } catch (err) {
      hud.toast(`Speed change failed: ${err.message}`, 'err');
    }
  };

  game.scene.add('FarmScene', scene, true, {
    initialState: state,
    catalog: state.catalog,
    selectedCrop: Object.keys(state.catalog)[0],
    hud,
    onAction,
    onPause,
    onSpeed,
  });

  const params = new URLSearchParams(window.location.search);
  const live = params.get('nolive') !== '1';

  if (live) {
    api.connect(
      (next) => scene.applyState(next),
      (status) => hud.setConnection(status),
    );
  } else {
    // Snapshot mode (?nolive=1): no stream and no polling, so headless
    // screenshot tools see an idle page.
    hud.setConnection('live');
  }

  window.addEventListener('beforeunload', () => {
    if (api.source) api.source.close();
  });

  window.farm = { api, scene, game, hud };
}

boot();
