/** DOM HUD: a thin, dumb renderer for the state produced by the server. */
export class Hud {
  constructor() {
    this.el = {
      clock: document.getElementById('hud-clock'),
      day: document.getElementById('hud-day'),
      weather: document.getElementById('hud-weather'),
      temp: document.getElementById('hud-temp'),
      forecast: document.getElementById('hud-forecast'),
      task: document.getElementById('hud-task'),
      objective: document.getElementById('hud-objective'),
      money: document.getElementById('hud-money'),
      moneyChip: document.getElementById('hud-money-chip'),
      moneyDelta: document.getElementById('hud-money-delta'),
      moneyWrap: document.getElementById('hud-money-wrap'),
      energy: document.getElementById('hud-energy'),
      energyBar: document.getElementById('hud-energy-bar'),
      can: document.getElementById('hud-can'),
      canBar: document.getElementById('hud-can-bar'),
      tank: document.getElementById('hud-tank'),
      tankBar: document.getElementById('hud-tank-bar'),
      seeds: document.getElementById('hud-seeds'),
      silo: document.getElementById('hud-silo'),
      siloBar: document.getElementById('hud-silo-bar'),
      robots: document.getElementById('hud-robots'),
      inspect: document.getElementById('hud-inspect'),
      events: document.getElementById('hud-events'),
      toast: document.getElementById('toast'),
      connection: document.getElementById('connection'),
    };
    this.lastEventsKey = '';
    this.lastMoney = null;
    this.moneyFlashTimer = null;
  }

  setConnection(status) {
    const node = this.el.connection;
    node.classList.remove('live', 'offline');
    if (status === 'live') {
      node.classList.add('live');
      node.textContent = 'live · server authoritative';
    } else {
      node.classList.add('offline');
      node.textContent = 'reconnecting…';
    }
  }

  update(state, extra = {}) {
    const { time, weather, player, tank, objective } = state;

    this.el.clock.textContent = time.paused ? `${time.clock} ⏸` : time.clock;
    this.el.day.textContent = `Day ${time.day} · ${time.season} · speed ${time.speed}x`;
    this.el.weather.textContent = weather.label;
    this.el.temp.textContent = `${weather.temp_c}°C`;

    const forecast = (weather.forecast || [])
      .map((entry) => entry.label || entry.type)
      .slice(0, 3)
      .join(' · ');
    this.el.forecast.textContent = forecast ? `next: ${forecast}` : '';

    this.el.task.textContent = state.task || '(no task)';

    const progress = Object.entries(objective.progress || {})
      .map(([key, value]) => `${key} ${value}`)
      .join(' · ');
    this.el.objective.textContent = objective.is_complete
      ? `DONE ${objective.label} (+$${objective.reward})`
      : objective.is_failed
        ? `FAILED ${objective.label} (deadline day ${objective.deadline_day})`
        : `${objective.label} — ${progress} (by day ${objective.deadline_day})`;
    this.el.objective.className = objective.is_complete
      ? 'objective done'
      : objective.is_failed
        ? 'objective failed'
        : 'objective';

    const money = player.money;
    this.el.money.textContent = `$${money.toFixed(2)}`;
    this.el.moneyChip.textContent = `$${money.toFixed(2)}`;
    if (this.lastMoney !== null && Math.abs(money - this.lastMoney) >= 0.005) {
      this.flashMoney(money - this.lastMoney);
    }
    this.lastMoney = money;
    this.el.energy.textContent = `${player.energy.toFixed(0)} / ${player.max_energy}`;
    this.el.energyBar.style.width = `${Math.max(0, Math.min(100, (player.energy / player.max_energy) * 100))}%`;

    this.el.can.textContent = `${player.can.litres.toFixed(1)} / ${player.can.capacity} L`;
    this.el.canBar.style.width = `${Math.min(100, (player.can.litres / player.can.capacity) * 100)}%`;

    this.el.tank.textContent = `${tank.litres.toFixed(0)} / ${tank.capacity} L`;
    this.el.tankBar.style.width = `${Math.min(100, tank.pct * 100)}%`;

    this.el.seeds.textContent = Object.entries(player.seeds)
      .map(([type, count]) => `${type.slice(0, 2)} ${count}`)
      .join(' · ');

    const silo = state.silo;
    const siloPct = silo.capacity_kg > 0 ? Math.min(1, silo.total_kg / silo.capacity_kg) : 0;
    const siloFull = silo.total_kg >= silo.capacity_kg - 0.05;
    this.el.silo.textContent = `${silo.total_kg.toFixed(1)} / ${silo.capacity_kg} kg`;
    this.el.silo.classList.toggle('warn', siloFull);
    this.el.siloBar.style.width = `${Math.round(siloPct * 100)}%`;
    this.el.siloBar.classList.toggle('full', siloFull);
    this.el.robots.textContent = state.robots.length
      ? state.robots.map((r) => `${r.id.replace('rover-', 'R')}:${r.status}`).join(' · ')
      : 'none';

    this.renderInspect(extra.plot, state);
    this.renderEvents(state.events);
    this.el.day.title = `score ${state.metrics.score} · tick ${state.tick}`;
  }

  /**
   * Brief +/- pop next to the clock. The chip text itself is rewritten on every
   * state frame, so the delta lives in its own badge and is restarted by
   * toggling the `show` class.
   */
  flashMoney(delta) {
    const wrap = this.el.moneyWrap;
    const badge = this.el.moneyDelta;
    if (!wrap || !badge) return;
    wrap.classList.remove('up', 'down');
    wrap.classList.add(delta > 0 ? 'up' : 'down');
    badge.textContent = `${delta > 0 ? '+' : '-'}$${Math.abs(delta).toFixed(2)}`;
    badge.classList.remove('show');
    void badge.offsetWidth; // restart the CSS animation
    badge.classList.add('show');
    clearTimeout(this.moneyFlashTimer);
    this.moneyFlashTimer = setTimeout(() => {
      wrap.classList.remove('up', 'down');
    }, 900);
  }

  renderInspect(plot) {
    if (!plot) {
      this.el.inspect.innerHTML = 'Stand on or next to a plot.';
      return;
    }
    const band = plot.band || 'good';
    const rows = [
      `<div class="row"><span>${plot.id}</span><span>${plot.tile[0]},${plot.tile[1]}</span></div>`,
      `<div class="row"><span>Moisture</span><span><span class="tag ${band}">${band}</span> ${(plot.moisture * 100).toFixed(0)}%</span></div>`,
      `<div class="row"><span>Soil</span><span>${plot.tilled ? 'tilled' : 'untilled'} · fert ${(plot.fertility * 100).toFixed(0)}%</span></div>`,
    ];
    if (plot.crop) {
      const crop = plot.crop;
      const stateLabel = crop.dead ? 'dead' : crop.ready ? 'ready' : crop.stage;
      rows.push(
        `<div class="row"><span>${crop.label}</span><span>${stateLabel}</span></div>`,
        `<div class="row"><span>Growth</span><span>${(crop.growth * 100).toFixed(0)}% · health ${(crop.health * 100).toFixed(0)}%</span></div>`,
      );
      if (crop.weeds > 0.01) {
        rows.push(`<div class="row"><span>Weeds</span><span>${(crop.weeds * 100).toFixed(0)}%</span></div>`);
      }
      if (crop.disease) rows.push(`<div class="row"><span>Disease</span><span>${crop.disease}</span></div>`);
    } else {
      rows.push('<div class="row"><span>Empty</span><span>till, then plant</span></div>');
    }
    rows.push(`<div class="row muted"><span>Distance</span><span>${plot.distance} tiles</span></div>`);
    this.el.inspect.innerHTML = rows.join('');
  }

  renderEvents(events = []) {
    const key = events.map((e) => `${e.tick}:${e.type}`).join('|');
    if (key === this.lastEventsKey) return;
    this.lastEventsKey = key;
    const items = events
      .slice(-6)
      .reverse()
      .map((event) => `<li title="day ${event.day}">${event.message}</li>`)
      .join('');
    this.el.events.innerHTML = items || '<li>No events yet.</li>';
  }

  toast(message, kind = 'ok') {
    if (!message) return;
    const node = document.createElement('div');
    node.className = `toast-item ${kind}`;
    node.textContent = message;
    this.el.toast.appendChild(node);
    setTimeout(() => {
      node.style.opacity = '0';
      node.style.transition = 'opacity 220ms ease';
      setTimeout(() => node.remove(), 240);
    }, 2200);
  }
}
