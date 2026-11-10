/**
 * Phaser renderer for the server-authoritative farm.
 *
 * The scene owns **no** game rules: it draws the last state it received and
 * sends intents to the server. Every visual is created once and then updated,
 * so streaming state cannot leak sprites.
 */

const TILE = 32;

const SOIL_DRY = 0x7a5c3a;
const SOIL_WET = 0x3f2a18;
const GRASS_TINT = 0x3d5f3a;

export class FarmScene extends Phaser.Scene {
  constructor() {
    super('FarmScene');
    this.state = null;
    this.plots = new Map();
    this.robotSprites = new Map();
    this.selectedCrop = 'wheat';
    this.selectedCropIndex = 0;
    this.showMoisture = false;
    this.pendingMove = null;
  }

  init(data = {}) {
    this.state = data.initialState || null;
    this.onAction = data.onAction || (() => {});
    this.hud = data.hud || null;
    this.catalog = data.catalog || (this.state ? this.state.catalog : {});
    this.onPause = data.onPause || null;
    this.onSpeed = data.onSpeed || null;
    if (data.selectedCrop) this.selectedCrop = data.selectedCrop;
  }

  preload() {
    this.load.spritesheet('tiles', 'assets/tiles.png', { frameWidth: 32, frameHeight: 32 });
    this.load.spritesheet('player', 'assets/player.png', { frameWidth: 48, frameHeight: 48 });
    this.load.image('stone', 'assets/stone.png');
    this.load.image('water_tank', 'assets/water_tank.png');
    this.load.spritesheet('waterani', 'assets/water2.png', { frameWidth: 48, frameHeight: 48 });
  }

  create() {
    this.buildGeneratedTextures();
    this.buildGround();
    this.buildField();
    this.buildProps();
    this.buildPlayer();
    this.buildOverlays();
    this.buildCamera();
    this.buildInput();

    // Render the state we booted with (also makes `?nolive=1` screenshots correct).
    this.applyState(this.state);

    this.events.once('shutdown', () => {
      this.plots.clear();
      this.robotSprites.clear();
    });

    // Diagnostics used by the headless smoke test (`npm run smoke`).
    document.body.dataset.scene = 'ready';
    document.body.dataset.plots = String(this.plots.size);
    document.body.dataset.world = `${this.state.world.w}x${this.state.world.h}`;
  }

  // -------------------------------------------------------------------------
  // Generated art (no extra asset files needed)
  // -------------------------------------------------------------------------
  buildGeneratedTextures() {
    const colorOf = (hex) => Phaser.Display.Color.HexStringToColor(hex).color;
    for (const [type, def] of Object.entries(this.catalog)) {
      for (let stage = 0; stage < 4; stage += 1) {
        this.makeCropTexture(`crop-${type}-${stage}`, stage, colorOf(def.color));
      }
    }
    this.makeRoverTexture('rover');
    this.makeBuildingTexture('silo', 76, 92, 0x9a7a52, 0x6d5233, 0xd9c7a3);
    this.makeBuildingTexture('farmhouse', 116, 96, 0xd9c7a3, 0x9b3d2f, 0x5b3a29);
    this.makePadTexture('charger', 0x1f4f68, 0x7ee0ff);
    this.makeDropTexture('drop');
  }

  makeCropTexture(key, stage, color) {
    if (this.textures.exists(key)) return;
    const g = this.make.graphics({ x: 0, y: 0, add: false });
    const baseY = 27;
    const stem = [10, 16, 22, 22][stage];
    const spread = [5, 8, 10, 10][stage];
    const cx = 13;

    // Stem
    g.fillStyle(0x2f6b35, 1).fillRect(cx - 1, baseY - stem, 3, stem);

    // Leaves
    g.fillStyle(0x3f8a45, 1);
    g.fillEllipse(cx - spread / 2 - 1, baseY - stem * 0.5, spread, 5);
    g.fillEllipse(cx + spread / 2 + 1, baseY - stem * 0.72, spread, 5);
    if (stage >= 2) {
      g.fillStyle(0x357a3c, 1);
      g.fillEllipse(cx - spread / 2, baseY - stem * 0.85, spread * 0.8, 4);
    }

    // Fruit / ear
    if (stage >= 2) {
      g.fillStyle(color, 1);
      g.fillEllipse(cx, baseY - stem - 3, 7, 8);
      if (stage === 3) {
        g.fillEllipse(cx - 5, baseY - stem, 5, 5);
        g.fillEllipse(cx + 5, baseY - stem, 5, 5);
        g.fillStyle(0xfff3c4, 0.85);
        g.fillEllipse(cx - 1, baseY - stem - 5, 2.5, 2.5);
      }
    }
    g.generateTexture(key, 26, 28);
    g.destroy();
  }

  makeRoverTexture(key) {
    if (this.textures.exists(key)) return;
    const g = this.make.graphics({ x: 0, y: 0, add: false });
    g.fillStyle(0x2b3a42, 1).fillRoundedRect(0, 4, 24, 14, 4);
    g.fillStyle(0x8ea8b5, 1).fillRoundedRect(2, 6, 14, 8, 2);
    g.fillStyle(0x6ee7a8, 1).fillRect(16, 8, 5, 4);
    g.fillStyle(0x11181d, 1).fillCircle(5, 19, 3).fillCircle(19, 19, 3);
    g.fillStyle(0xffd166, 1).fillRect(11, 0, 2, 5);
    g.generateTexture(key, 24, 22);
    g.destroy();
  }

  makeBuildingTexture(key, w, h, wall, roof, trim) {
    if (this.textures.exists(key)) return;
    const g = this.make.graphics({ x: 0, y: 0, add: false });
    g.fillStyle(0x000000, 0.18).fillEllipse(w / 2, h - 4, w * 0.92, 10);
    g.fillStyle(wall, 1).fillRoundedRect(4, h * 0.34, w - 8, h * 0.62, 3);
    g.fillStyle(roof, 1).fillTriangle(0, h * 0.4, w, h * 0.4, w / 2, 2);
    g.fillStyle(trim, 1).fillRect(w / 2 - 8, h * 0.6, 16, h * 0.36);
    g.fillStyle(0x2c3e50, 0.75).fillRect(10, h * 0.46, 10, 8).fillRect(w - 20, h * 0.46, 10, 8);
    g.generateTexture(key, w, h);
    g.destroy();
  }

  makePadTexture(key, base, glyph) {
    if (this.textures.exists(key)) return;
    const g = this.make.graphics({ x: 0, y: 0, add: false });
    g.fillStyle(base, 1).fillRoundedRect(2, 2, 28, 28, 5);
    g.lineStyle(1, glyph, 0.8).strokeRoundedRect(2, 2, 28, 28, 5);
    g.fillStyle(glyph, 1);
    g.fillTriangle(17, 7, 11, 17, 16, 17);
    g.fillTriangle(15, 25, 21, 15, 16, 15);
    g.generateTexture(key, 32, 32);
    g.destroy();
  }

  makeDropTexture(key) {
    if (this.textures.exists(key)) return;
    const g = this.make.graphics({ x: 0, y: 0, add: false });
    g.fillStyle(0x4cc9f0, 0.95);
    g.fillCircle(5, 8, 4);
    g.fillTriangle(1, 8, 9, 8, 5, 0);
    g.generateTexture(key, 10, 13);
    g.destroy();
  }

  // -------------------------------------------------------------------------
  // World layers
  // -------------------------------------------------------------------------
  buildGround() {
    const { w, h, px_w: pxW, px_h: pxH } = this.state.world;
    this.worldPx = { w: pxW, h: pxH };

    this.ground = this.add.renderTexture(0, 0, pxW, pxH).setOrigin(0, 0).setDepth(-20);

    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        let frame = 0;
        if ((x * 7 + y * 13) % 19 === 0) frame = 2;
        else if ((x * 5 + y * 11) % 23 === 0) frame = 1;
        this.ground.drawFrame('tiles', frame, x * TILE, y * TILE);
      }
    }

    // The cultivated field reads as one plot of land even before tilling.
    for (const plot of this.state.plots) {
      this.ground.drawFrame('tiles', 5, plot.tile[0] * TILE, plot.tile[1] * TILE, 0.5);
    }
  }

  buildField() {
    for (const plot of this.state.plots) {
      const [tx, ty] = plot.tile;
      const x = tx * TILE;
      const y = ty * TILE;

      const soil = this.add
        .rectangle(x + 1, y + 1, TILE - 2, TILE - 2, SOIL_DRY, 0.9)
        .setOrigin(0, 0)
        .setStrokeStyle(1, 0x1a1208, 0.25)
        .setDepth(-10);

      const crop = this.add
        .image(x + TILE / 2, y + TILE - 2, 'crop-wheat-0')
        .setOrigin(0.5, 1)
        .setDepth(1)
        .setVisible(false);

      const dew = this.add
        .image(x + TILE - 7, y + 7, 'drop')
        .setDepth(3)
        .setScale(0.8)
        .setAlpha(0.9)
        .setVisible(false);
      const ready = this.add
        .circle(x + TILE - 6, y + 6, 4, 0xffd166)
        .setDepth(3)
        .setVisible(false);

      this.plots.set(plot.id, { soil, crop, dew, ready, tile: plot.tile });
    }
  }

  buildProps() {
    for (const obstacle of this.state.obstacles) {
      this.add.image(obstacle.pos[0], obstacle.pos[1], 'stone').setDepth(4).setScale(0.8);
    }

    const tank = this.state.tank;
    this.tankSprite = this.add
      .image(tank.pos[0], tank.pos[1] + TILE / 2, 'water_tank')
      .setOrigin(0.5, 1)
      .setDepth(6);
    const tankScale = (TILE * 2.1) / (this.tankSprite.width || TILE * 2);
    this.tankSprite.setScale(tankScale);

    this.tankFill = this.add.rectangle(tank.pos[0], tank.pos[1] + TILE / 2, 18, 4, 0x4cc9f0, 0.9).setDepth(7);
    this.tankFill.setVisible(false);

    const silo = this.state.landmarks.silo;
    const siloX = silo[0] * TILE + TILE / 2;
    const siloBottom = silo[1] * TILE + TILE;
    this.add
      .image(siloX, siloBottom, 'silo')
      .setOrigin(0.5, 1)
      .setDepth(6)
      .setScale(0.9);

    // Fill meter sits just below the barn: the tile row above it is covered by
    // the top-left HUD panel, so a label there would often be hidden.
    this.add.rectangle(siloX, siloBottom + 6, 58, 8, 0x0b1116, 0.85).setDepth(8);
    this.siloFill = this.add
      .rectangle(siloX - 27, siloBottom + 6, 0, 6, 0xf0c674, 1)
      .setOrigin(0, 0.5)
      .setDepth(9);
    this.siloLabel = this.add
      .text(siloX, siloBottom + 11, '0.0 / 60 kg', {
        fontFamily: 'Segoe UI, sans-serif',
        fontSize: '10px',
        color: '#ffe0a3',
      })
      .setOrigin(0.5, 0)
      .setDepth(9);

    const house = this.state.landmarks.farmhouse;
    this.add
      .image(house[0] * TILE + TILE / 2, house[1] * TILE + TILE, 'farmhouse')
      .setOrigin(0.5, 1)
      .setDepth(6)
      .setScale(1);

    const charger = this.state.landmarks.charger;
    this.add
      .image(charger[0] * TILE + TILE / 2, charger[1] * TILE + TILE / 2, 'charger')
      .setDepth(3);

    this.add
      .text(tank.pos[0], tank.pos[1] - 70, 'WATER', {
        fontFamily: 'Segoe UI, sans-serif',
        fontSize: '10px',
        color: '#9fd8ef',
      })
      .setOrigin(0.5)
      .setDepth(8);
  }

  buildPlayer() {
    const player = this.state.player;
    this.player = this.add
      .sprite(player.pos[0], player.pos[1] + 8, 'player', 0)
      .setOrigin(0.5, 0.8)
      .setDepth(10)
      .setScale(0.95);

    this.createAnims();
    this.player.play(`idle-${player.facing || 'down'}`);
    this.playerTile = [...player.tile];
  }

  createAnims() {
    const directions = {
      down: [0, 4, 8, 12],
      left: [1, 5, 9, 13],
      right: [3, 7, 11, 15],
      up: [2, 6, 10, 14],
    };
    for (const [dir, frames] of Object.entries(directions)) {
      if (!this.anims.exists(`walk-${dir}`)) {
        this.anims.create({
          key: `walk-${dir}`,
          frames: this.anims.generateFrameNumbers('player', { frames }),
          frameRate: 7,
          repeat: -1,
        });
      }
      if (!this.anims.exists(`idle-${dir}`)) {
        this.anims.create({
          key: `idle-${dir}`,
          frames: [{ key: 'player', frame: frames[0] }],
          frameRate: 1,
          repeat: -1,
        });
      }
    }
  }

  buildOverlays() {
    if (!this.textures.exists('rain')) {
      const g = this.make.graphics({ x: 0, y: 0, add: false });
      g.lineStyle(1, 0xbfe3ff, 0.45);
      for (let i = 0; i < 9; i += 1) {
        g.beginPath();
        g.moveTo(i * 8, 0);
        g.lineTo(i * 8 - 10, 64);
        g.strokePath();
      }
      g.generateTexture('rain', 64, 64);
      g.destroy();
    }

    // Unit-sized tints are scaled to the camera's world view every frame.
    this.nightOverlay = this.add.rectangle(0, 0, 1, 1, 0x0a1b3a, 0).setDepth(60);
    this.weatherOverlay = this.add.rectangle(0, 0, 1, 1, 0x5a7fa8, 0).setDepth(61);
    this.rainLayer = this.add.tileSprite(0, 0, 64, 64, 'rain').setDepth(62).setAlpha(0);
  }

  /** Size and centre the weather tints on the visible world rectangle. */
  syncOverlays() {
    const cam = this.cameras.main;
    const view = cam.worldView;
    if (!view || !this.nightOverlay) return;
    const width = view.width + 8;
    const height = view.height + 8;

    this.nightOverlay.setPosition(view.centerX, view.centerY).setScale(width, height);
    this.weatherOverlay.setPosition(view.centerX, view.centerY).setScale(width, height);
    this.rainLayer.setPosition(view.centerX, view.centerY).setSize(width, height);
  }

  buildCamera() {
    const cam = this.cameras.main;
    cam.setBounds(0, 0, this.worldPx.w, this.worldPx.h);
    cam.startFollow(this.player, true, 0.12, 0.12);
    cam.setZoom(1.6);
    cam.setRoundPixels(true);

    this.input.on('wheel', (pointer, objects, dx, dy) => {
      const next = Phaser.Math.Clamp(cam.zoom - Math.sign(dy) * 0.15, 0.9, 2.6);
      cam.zoomTo(next, 120);
    });
  }

  buildInput() {
    const kb = this.input.keyboard;
    const on = (event, fn) => kb.on(event, fn);

    on('keydown-UP', () => this.step('up'));
    on('keydown-DOWN', () => this.step('down'));
    on('keydown-LEFT', () => this.step('left'));
    on('keydown-RIGHT', () => this.step('right'));
    on('keydown-W', () => this.step('up'));
    on('keydown-S', () => this.step('down'));
    on('keydown-A', () => this.step('left'));
    on('keydown-D', () => this.step('right'));

    on('keydown-SPACE', () => this.contextAction());
    on('keydown-C', () => this.send({ type: 'collect_water' }));
    on('keydown-R', () => this.send({ type: 'rest' }));
    on('keydown-M', () => this.toggleMoisture());
    on('keydown-F', () => this.send({ type: 'dispatch', task: 'water' }));
    on('keydown-G', () => this.sellAll());
    on('keydown-P', () => this.onPause && this.onPause());
    on('keydown-PLUS', () => this.onSpeed && this.onSpeed(1));
    on('keydown-MINUS', () => this.onSpeed && this.onSpeed(-1));
    on('keydown-ONE', () => this.selectCrop(0));
    on('keydown-TWO', () => this.selectCrop(1));
    on('keydown-THREE', () => this.selectCrop(2));
    on('keydown-FOUR', () => this.selectCrop(3));
  }

  // -------------------------------------------------------------------------
  // Intents (never rules)
  // -------------------------------------------------------------------------
  send(action) {
    return this.onAction(action);
  }

  step(direction) {
    const now = Date.now();
    if (now - (this.lastStepAt || 0) < 80) return;
    this.lastStepAt = now;
    this.send({ type: 'move', dir: direction });
  }

  selectCrop(index) {
    const types = Object.keys(this.catalog);
    const type = types[index];
    if (!type) return;
    this.selectedCrop = type;
    this.hud && this.hud.toast(`Selected ${this.catalog[type].label} seeds`, 'ok');
  }

  sellAll() {
    if (!this.state) return;
    const crops = Object.keys(this.state.silo.items || {});
    if (!crops.length) {
      this.hud && this.hud.toast('Silo is empty', 'err');
      return;
    }
    for (const crop of crops) this.send({ type: 'sell', crop });
  }

  toggleMoisture() {
    this.showMoisture = !this.showMoisture;
    for (const entry of this.plots.values()) {
      if (!entry.label) continue;
      entry.label.setVisible(this.showMoisture);
    }
  }

  contextAction() {
    if (!this.state) return;
    const tile = this.targetTile();
    const plot = this.plotAt(tile);
    if (!plot) {
      this.send({ type: 'inspect', tile: [tile.x, tile.y] });
      return;
    }
    const crop = plot.crop;
    if (crop && crop.dead) return this.send({ type: 'clear', tile: plot.tile });
    if (crop && crop.ready) return this.send({ type: 'harvest', tile: plot.tile });
    if (crop && crop.weeds > 0.3) return this.send({ type: 'weed', tile: plot.tile });
    if (crop && crop.disease) return this.send({ type: 'treat', tile: plot.tile });
    if (crop) return this.send({ type: 'water', tile: plot.tile });
    if (!plot.tilled) return this.send({ type: 'till', tile: plot.tile });
    return this.send({ type: 'plant_crop', crop: this.selectedCrop, tile: plot.tile });
  }

  // -------------------------------------------------------------------------
  // State application (idempotent: every frame is derived from `state`)
  // -------------------------------------------------------------------------
  applyState(state) {
    if (!state) return;
    this.state = state;
    this.syncSilo(state);
    if (!this.plots.size) {
      if (this.hud) this.hud.update(state, {});
      return;
    }
    this.updatePlayer(state);
    this.updatePlots(state);
    this.updateRobots(state);
    this.updateWeather(state);
    const extra = { plot: this.inspectPlot() };
    if (this.hud) this.hud.update(state, extra);
    this.events.emit('state', state, extra);
  }

  /** Kg meter under the barn; turns red once the silo is (nearly) full. */
  syncSilo(state) {
    if (!this.siloFill || !this.siloLabel || !state.silo) return;
    const { total_kg: kg, capacity_kg: capacity } = state.silo;
    const pct = capacity > 0 ? Phaser.Math.Clamp(kg / capacity, 0, 1) : 0;
    this.siloFill.setSize(Math.round(54 * pct), 6);
    this.siloFill.setFillStyle(pct >= 0.95 ? 0xff6b6b : 0xf0c674, 1);
    this.siloLabel.setText(`${kg.toFixed(1)} / ${capacity} kg`);
    this.siloLabel.setColor(pct >= 0.95 ? '#ffb0b0' : '#ffe0a3');
  }

  /**
   * Length of the walk tween for a move of `tiles` tiles.
   *
   * `move` reports a single tile, but `move_to` reports the whole A* path in one
   * frame, so the sprite has to cover the reported distance in a single tween.
   * A fixed 150 ms turns a long `move_to` into a teleport, so the duration grows
   * with the distance (~60 ms per tile) and is capped, so the sprite never lags
   * far behind the server.
   */
  walkDuration(tiles) {
    const distance = Math.max(1, Number(tiles) || 1);
    return Math.min(Math.max(150, distance * 60), 1200);
  }

  updatePlayer(state) {
    const player = state.player;
    const targetX = player.pos[0];
    const targetY = player.pos[1] + 8;
    const prevTile = this.playerTile;
    const stepX = player.tile[0] - prevTile[0];
    const stepY = player.tile[1] - prevTile[1];
    const tiles = Math.abs(stepX) + Math.abs(stepY);
    const moved = tiles > 0;
    // `move_to` does not set `facing`, so a multi-tile frame faces the way the
    // sprite actually travels instead of wherever it was looking before.
    let direction = player.facing || 'down';
    if (moved) {
      direction = Math.abs(stepX) >= Math.abs(stepY)
        ? (stepX > 0 ? 'right' : 'left')
        : (stepY > 0 ? 'down' : 'up');
    }

    if (moved) {
      this.playerTile = [...player.tile];
      this.tweens.killTweensOf(this.player);
      this.player.play(`walk-${direction}`, true);
      this.tweens.add({
        targets: this.player,
        x: targetX,
        y: targetY,
        duration: this.walkDuration(tiles),
        ease: 'Linear',
        onComplete: () => this.player.play(`idle-${direction}`, true),
      });
    } else if (Math.abs(this.player.x - targetX) > 1 || Math.abs(this.player.y - targetY) > 1) {
      // Same tile but a different pixel origin (e.g. after a reset): glide just
      // as long as the distance asks for instead of snapping.
      const drift = Math.max(
        Math.abs(this.player.x - targetX),
        Math.abs(this.player.y - targetY),
      ) / TILE;
      this.tweens.killTweensOf(this.player);
      this.tweens.add({
        targets: this.player,
        x: targetX,
        y: targetY,
        duration: this.walkDuration(drift),
      });
    }
  }

  updatePlots(state) {
    for (const plot of state.plots) {
      const entry = this.plots.get(plot.id);
      if (!entry) continue;

      if (!plot.known) {
        entry.soil.setFillStyle(0x0c1216, 0.9);
        entry.crop.setVisible(false);
        entry.dew.setVisible(false);
        entry.ready.setVisible(false);
        if (entry.label) entry.label.setVisible(false);
        continue;
      }

      const wet = Phaser.Display.Color.Interpolate.ColorWithColor(
        Phaser.Display.Color.ValueToColor(SOIL_DRY),
        Phaser.Display.Color.ValueToColor(SOIL_WET),
        100,
        Math.round(plot.moisture * 100),
      );
      entry.soil.setFillStyle(
        Phaser.Display.Color.GetColor(wet.r, wet.g, wet.b),
        plot.tilled ? 0.95 : 0.28,
      );

      const crop = plot.crop;
      const showCrop = Boolean(crop && !crop.dead);
      entry.crop.setVisible(showCrop);
      if (showCrop) {
        const stage = crop.ready ? 3 : Math.max(0, Math.min(2, crop.stage_index ?? 1));
        const key = `crop-${crop.type}-${stage}`;
        if (this.textures.exists(key)) entry.crop.setTexture(key);
        entry.crop.setScale(0.95 + stage * 0.12);
        entry.crop.setAlpha(crop.health < 0.45 ? 0.6 : 1);
        entry.crop.setTint(crop.disease ? 0xc9d8a0 : 0xffffff);
      }

      entry.dew.setVisible(Boolean(crop && !crop.dead && plot.needs_water));
      entry.ready.setVisible(Boolean(crop && crop.ready));

      if (!entry.label) {
        entry.label = this.add
          .text(plot.tile[0] * TILE + 2, plot.tile[1] * TILE + 1, '', {
            fontFamily: 'Segoe UI, sans-serif',
            fontSize: '9px',
            color: '#e8f7ff',
            backgroundColor: 'rgba(0,0,0,0.45)',
            padding: { x: 2, y: 1 },
          })
          .setDepth(5);
      }
      entry.label.setText(`${Math.round(plot.moisture * 100)}`);
      entry.label.setVisible(this.showMoisture);
    }

    const tank = state.tank;
    if (this.tankFill) {
      this.tankFill.setScale(Math.max(0.05, tank.pct), 1);
      this.tankFill.setFillStyle(tank.pct < 0.2 ? 0xff8f6b : 0x4cc9f0, 0.9);
    }
  }

  updateRobots(state) {
    const seen = new Set();
    for (const robot of state.robots) {
      seen.add(robot.id);
      let sprite = this.robotSprites.get(robot.id);
      if (!sprite) {
        sprite = this.add.image(robot.pos[0], robot.pos[1], 'rover').setDepth(9);
        sprite.label = this.add
          .text(robot.pos[0], robot.pos[1] - 18, robot.id.replace('rover-', 'R'), {
            fontFamily: 'Segoe UI, sans-serif',
            fontSize: '9px',
            color: '#6ee7a8',
          })
          .setOrigin(0.5)
          .setDepth(9);
        this.robotSprites.set(robot.id, sprite);
      }
      this.tweens.add({ targets: sprite, x: robot.pos[0], y: robot.pos[1], duration: 320 });
      if (sprite.label) sprite.label.setPosition(robot.pos[0], robot.pos[1] - 18);
      sprite.setAlpha(robot.battery < 0.2 ? 0.55 : 1);
    }

    for (const [id, sprite] of this.robotSprites) {
      if (seen.has(id)) continue;
      if (sprite.label) sprite.label.destroy();
      sprite.destroy();
      this.robotSprites.delete(id);
    }
  }

  updateWeather(state) {
    const minutes = state.time.minute;
    const night = minutes < 6 * 60 || minutes >= 20 * 60;
    const dusk = !night && (minutes < 7 * 60 || minutes > 19 * 60);
    const alpha = night ? 0.42 : dusk ? 0.16 : 0;
    this.nightOverlay.setFillStyle(0x0a1b3a, alpha);

    const type = state.weather.type;
    const rainAlpha = type === 'storm' ? 0.55 : type === 'rain' ? 0.32 : 0;
    this.weatherOverlay.setFillStyle(0x3d5a75, rainAlpha * 0.3);
    this.rainLayer.setAlpha(rainAlpha);
  }

  update(time, delta) {
    this.syncOverlays();
    if (this.rainLayer && this.rainLayer.alpha > 0) {
      this.rainLayer.tilePositionY -= delta * 0.5;
      this.rainLayer.tilePositionX -= delta * 0.08;
    }
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------
  plotAt(tile) {
    if (!this.state) return null;
    return this.state.plots.find((p) => p.tile[0] === tile.x && p.tile[1] === tile.y) || null;
  }

  targetTile() {
    if (!this.state) return { x: 0, y: 0 };
    const [x, y] = this.state.player.tile;
    if (this.plotAt({ x, y })) return { x, y };
    const vectors = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
    const [dx, dy] = vectors[this.state.player.facing || 'down'];
    return { x: x + dx, y: y + dy };
  }

  inspectPlot() {
    const plot = this.plotAt(this.targetTile());
    return plot && plot.known ? plot : null;
  }
}
