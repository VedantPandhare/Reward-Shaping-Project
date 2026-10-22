// One arena panel: its own WebGL renderer + bloom, 3D scene, lander visuals, HUD and reward chart.
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { W, H, MAX_STEPS } from "./lander_env.js";
import { Particles, buildLander, buildSky, buildTerrain } from "./scene.js";

const TERM_COLORS = {
  distance: "#4cc9f0", speed: "#f8961e", tilt: "#90be6d", legs: "#b388eb", fuel: "#f94144",
  terminal: "#adb5bd", hover_bonus: "#ff2bd6", tilt_bonus: "#ff2bd6", delta_shaping: "#ffd60a",
  time_penalty: "#8d99ae", extra_fuel: "#e5383b", touchdown_penalty: "#ff7b00",
};
const BONUS_TERMS = new Set(["hover_bonus", "tilt_bonus"]);
export const TERM_LABELS = {
  hover_bonus: "hover bonus", tilt_bonus: "tilt bonus", delta_shaping: "delta shaping",
  time_penalty: "time penalty", extra_fuel: "extra fuel", touchdown_penalty: "touchdown penalty",
};
const GROUP = 5; // frames per chart bar
const CHART_BARS = 70;

const smooth = (a, b, k) => a + (b - a) * k;

function snapshot(env) {
  const p = env.lander.getPosition();
  return {
    x: p.x, y: p.y, a: env.lander.getAngle(),
    legs: env.legs.map((l) => ({ x: l.getPosition().x, y: l.getPosition().y, a: l.getAngle() })),
  };
}

export class Panel {
  constructor(root) {
    this.root = root;
    this.$ = (sel) => root.querySelector(sel);
    this.canvas = this.$("canvas.stage");
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.NoToneMapping;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x02030a);
    this.camera = new THREE.PerspectiveCamera(38, 1.6, 0.1, 400);
    this.sky = buildSky(this.scene);

    this.scene.add(new THREE.HemisphereLight(0x6f86c9, 0x1a1c24, 0.55));
    this.sun = new THREE.DirectionalLight(0xfff0dc, 3.2);
    this.sun.position.set(-8, 16, 14);
    this.sun.target.position.set(W / 2, 3, 0);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    Object.assign(this.sun.shadow.camera, { left: -14, right: 14, top: 12, bottom: -6, near: 1, far: 60 });
    this.sun.shadow.bias = -0.0004;
    this.scene.add(this.sun, this.sun.target);

    this.particles = new Particles();
    this.scene.add(this.particles.points);

    this.lander = buildLander(0x4d8dff);
    this.scene.add(this.lander.root);

    const trailGeo = new THREE.BufferGeometry();
    this.trailPos = new Float32Array((MAX_STEPS + 2) * 3);
    trailGeo.setAttribute("position", new THREE.BufferAttribute(this.trailPos, 3));
    trailGeo.setDrawRange(0, 0);
    this.trailMat = new THREE.LineBasicMaterial({ color: 0x4d8dff, transparent: true, opacity: 0.55 });
    this.trail = new THREE.Line(trailGeo, this.trailMat);
    this.trail.frustumCulled = false;
    this.scene.add(this.trail);

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(512, 512), 0.55, 0.6, 0.82);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.chart = this.$("canvas.chart");
    this.cctx = this.chart.getContext("2d");
    this.terrainGroup = null;
    this.camState = { x: W / 2, d: 0, shake: 0 };
    this.hudTimer = 0;
    this.resize();
  }

  resize() {
    const r = this.canvas.getBoundingClientRect();
    const w = Math.max(200, Math.floor(r.width)), h = Math.max(150, Math.floor(r.height));
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.particles.uniforms.uScale.value = h * 0.9 * this.renderer.getPixelRatio();
    const cr = this.chart.getBoundingClientRect();
    this.chart.width = Math.floor(cr.width * devicePixelRatio);
    this.chart.height = Math.floor(cr.height * devicePixelRatio);
    this.baseDist = this.fitDistance(W + 2.5);
  }

  fitDistance(width) {
    const vfov = (this.camera.fov * Math.PI) / 180;
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * this.camera.aspect);
    const dW = width / 2 / Math.tan(hfov / 2);
    const dH = (H + 3.4) / 2 / Math.tan(vfov / 2);
    return Math.max(dW, dH);
  }

  /** Start a new episode with a freshly generated terrain. */
  setup(env, driver) {
    this.env = env;
    this.driver = driver;
    if (this.terrainGroup) {
      this.scene.remove(this.terrainGroup);
      this.terrainGroup.traverse((o) => { o.geometry?.dispose(); });
    }
    const helipad = { x1: env.helipadX1, x2: env.helipadX2, y: env.helipadY };
    const t = buildTerrain(env.terrain, helipad);
    this.terrainGroup = t.group;
    this.heightAt = t.heightAt;
    this.beacons = t.beacons;
    this.scene.add(t.group);
    this.lander.tint(driver.color);
    this.trailMat.color.set(driver.color);
    this.cur = snapshot(env);
    this.prev = this.cur;
    this.trailCount = 0;
    this.trail.geometry.setDrawRange(0, 0);
    this.steps = [];
    this.flame = { main: false, left: false, right: false };
    this.dead = false;
    this.camState = { x: W / 2, d: this.baseDist, shake: 0 };
    this.lastTerms = {};
    this.statusEl = this.$(".status");
    this.statusEl.className = "status";
    this.statusEl.textContent = "";
    this.$(".drv-name").textContent = driver.label;
    this.$(".drv-blurb").textContent = driver.blurb;
    this.root.style.setProperty("--accent", driver.cssColor);
    this.$(".hackbox").style.display = driver.hasBonus ? "" : "none";
    this.$(".score-shaped").style.display = driver.human ? "none" : "";
    for (const b of this.beacons) b.material.color.set(b.position.y > helipad.y + 1 ? 0x40ffb0 : 0xff5a3c);
    this.drawChart();
    this.updateHud(true);
  }

  tick(result, action) {
    const env = this.env;
    this.prev = this.cur;
    this.cur = snapshot(env);
    this.flame = { ...env.flame, action };
    this.steps.push(result.terms);
    this.lastTerms = result.terms;
    // trail
    const i = this.trailCount++;
    this.trailPos.set([this.cur.x, this.cur.y, 0.45], i * 3);
    this.trail.geometry.attributes.position.needsUpdate = true;
    this.trail.geometry.setDrawRange(0, this.trailCount);

    const l = this.cur, ca = Math.cos(l.a), sa = Math.sin(l.a);
    const alt = l.y - this.heightAt(l.x);
    if (env.flame.main) {
      const nx = l.x + sa * 0.55, ny = l.y - ca * 0.55; // nozzle: "down" in the lander frame
      for (let k = 0; k < 2; k++) {
        const sp = 3 + Math.random() * 2;
        this.particles.emit(nx, ny, 0, sa * sp + (Math.random() - 0.5) * 1.2, -ca * sp, (Math.random() - 0.5) * 0.8,
          0.45 + Math.random() * 0.3, 0.55 + Math.random() * 0.35, [1, 0.55 + Math.random() * 0.3, 0.12], 2.2);
      }
      if (alt < 3.2) {
        for (let k = 0; k < 3; k++) {
          const dir = Math.random() < 0.5 ? -1 : 1;
          this.particles.emit(l.x, this.heightAt(l.x) + 0.1, (Math.random() - 0.5) * 1.5, dir * (2 + Math.random() * 3), 0.3 + Math.random() * 0.8, (Math.random() - 0.5) * 3,
            0.9 + Math.random() * 0.6, 0.7 + Math.random() * 0.5, [0.34, 0.31, 0.28], 1.6);
        }
      }
    }
    if (env.flame.left || env.flame.right) {
      const dir = env.flame.left ? -1 : 1;
      this.particles.emit(l.x + dir * 0.75 * ca, l.y + 0.2, 0, dir * (2 + Math.random()), (Math.random() - 0.5) * 0.6, 0, 0.25, 0.3, [1, 0.7, 0.3], 3);
    }
    if (result.outcome) this.finish(result.outcome);
  }

  finish(outcome) {
    this.dead = true;
    const l = this.cur;
    const raw = this.env.returns.raw;
    if (outcome === "crashed") {
      for (let k = 0; k < 110; k++) {
        const a = Math.random() * Math.PI * 2, s = 1 + Math.random() * 6;
        this.particles.emit(l.x, l.y, 0, Math.cos(a) * s, Math.sin(a) * s + 2, (Math.random() - 0.5) * 4,
          0.7 + Math.random() * 1.1, 0.9 + Math.random() * 0.9,
          Math.random() < 0.6 ? [1, 0.45 + Math.random() * 0.4, 0.1] : [0.5, 0.5, 0.55], 1.1);
      }
      this.camState.shake = 1;
      this.lander.tint(0x2a2a2e);
      this.statusEl.textContent = `CRASHED  ${raw.toFixed(0)}`;
    } else if (outcome === "landed") {
      for (let k = 0; k < 30; k++) {
        this.particles.emit(l.x, this.heightAt(l.x) + 0.1, (Math.random() - 0.5), (Math.random() < 0.5 ? -1 : 1) * (1 + Math.random() * 2), 0.5 + Math.random(), (Math.random() - 0.5) * 2,
          1 + Math.random(), 0.8, [0.3, 0.9, 0.6], 1.5);
      }
      this.statusEl.textContent = `LANDED  +${raw.toFixed(0)}`;
    } else {
      this.statusEl.textContent = `TIMED OUT  ${raw.toFixed(0)}  - never landed`;
    }
    this.statusEl.className = `status show ${outcome}`;
    this.updateHud(true);
  }

  render(dt, time, alpha) {
    const a = Math.min(Math.max(alpha, 0), 1);
    const c = this.cur, p = this.prev, L = this.lander;
    const x = smooth(p.x, c.x, a), y = smooth(p.y, c.y, a), ang = smooth(p.a, c.a, a);
    L.body.rotation.z = ang;
    L.body.position.set(x, y, 0);
    const R = (v) => ({ x: x + v[0] * Math.cos(ang) - v[1] * Math.sin(ang), y: y + v[0] * Math.sin(ang) + v[1] * Math.cos(ang) });
    for (let i = 0; i < 2; i++) {
      const pl = p.legs[i], cl = c.legs[i];
      const lx = smooth(pl.x, cl.x, a), ly = smooth(pl.y, cl.y, a), la = smooth(pl.a, cl.a, a);
      L.legs[i].position.set(lx, ly, 0);
      L.legs[i].rotation.z = la;
      const side = lx > x ? 1 : -1;
      const top = new THREE.Vector3(lx - Math.sin(la) * 0.27, ly + Math.cos(la) * 0.27, 0);
      const att = R([side * 0.4, -0.1]);
      L.placeStrut(L.struts[i], new THREE.Vector3(att.x, att.y, 0), top);
    }
    // flames
    const f = this.flame || {};
    const flick = 0.85 + 0.3 * Math.sin(time * 90 + Math.random());
    L.mainFlame.visible = !!f.main && !this.dead;
    L.mainFlame.scale.set(flick, 0.8 + 0.5 * Math.random(), flick);
    L.sideFlames[0].visible = !!f.left && !this.dead;
    L.sideFlames[1].visible = !!f.right && !this.dead;
    L.light.intensity = f.main && !this.dead ? 22 + 8 * Math.random() : 0;
    for (const b of this.beacons) b.scale.setScalar(1 + 0.25 * Math.sin(time * 4 + b.position.x));
    this.sky.earth.rotation.y += dt * 0.02;

    this.particles.update(dt);
    this.updateCamera(dt, x, y);
    this.composer.render(dt);

    this.hudTimer += dt;
    if (this.hudTimer > 0.05) { this.hudTimer = 0; this.updateHud(false); }
  }

  updateCamera(dt, x, y) {
    const cs = this.camState;
    const alt = y - this.heightAt(x);
    const follow = THREE.MathUtils.smoothstep(6.5 - alt, 0, 4) * (this.dead ? 0.6 : 1);
    const targetX = smooth(W / 2, THREE.MathUtils.clamp(x, 4, W - 4), follow * 0.85);
    const targetD = this.baseDist * (1 - 0.42 * follow);
    const k = 1 - Math.exp(-dt * 3);
    cs.x = smooth(cs.x, targetX, k); cs.d = smooth(cs.d, targetD, k);
    const ty = smooth(H / 2 - 0.2, Math.max(y, this.env.helipadY + 1.2) - 0.2, follow * 0.8);
    cs.shake = Math.max(0, cs.shake - dt * 1.6);
    const sx = (Math.random() - 0.5) * cs.shake * 0.5, sy = (Math.random() - 0.5) * cs.shake * 0.5;
    this.camera.position.set(cs.x + sx + (x - W / 2) * 0.02, ty + 0.9 + sy, cs.d);
    this.camera.lookAt(cs.x + sx, ty - 0.1, 0);
  }

  // ---------- HUD ----------
  updateHud(force) {
    const env = this.env;
    if (!env) return;
    const r = env.returns, o = env.obs;
    const alt = Math.max(0, this.cur.y - this.env.helipadY - 0.6);
    const set = (sel, v) => { const e = this.$(sel); if (e.textContent !== v) e.textContent = v; };
    set(".s-frame", `${env.t}/${MAX_STEPS}`);
    set(".s-alt", alt.toFixed(1));
    set(".s-speed", Math.hypot(o[2], o[3]).toFixed(2));
    set(".s-fuel", r.fuel.toFixed(1));
    set(".s-raw", r.raw.toFixed(0));
    set(".s-shaped", r.shaped.toFixed(0));
    this.$(".bar-frame").style.width = `${(env.t / MAX_STEPS) * 100}%`;
    if (this.driver.hasBonus) {
      const share = r.absTotal > 0 ? Math.abs(r.custom) / r.absTotal : 0;
      this.$(".bar-share").style.width = `${Math.min(share, 1) * 100}%`;
      set(".s-share", `${(share * 100).toFixed(0)}%`);
      const hacked = share > 0.5 && env.t > 400;
      this.$(".hackalert").classList.toggle("on", hacked);
    }
    this.drawChart();
  }

  drawChart() {
    const ctx = this.cctx, cv = this.chart;
    const w = cv.width, h = cv.height;
    ctx.clearRect(0, 0, w, h);
    const groups = [];
    for (let i = 0; i < this.steps.length; i += GROUP) {
      const acc = {};
      for (const t of this.steps.slice(i, i + GROUP)) for (const [k, v] of Object.entries(t)) if (k !== "env") acc[k] = (acc[k] || 0) + v;
      groups.push(acc);
    }
    const shown = groups.slice(-CHART_BARS);
    let maxPos = 4, maxNeg = 4;
    for (const g of shown) {
      let pos = 0, neg = 0;
      for (const [k, v] of Object.entries(g)) { if (k === "terminal") continue; if (v > 0) pos += v; else neg += v; }
      maxPos = Math.max(maxPos, pos); maxNeg = Math.max(maxNeg, -neg);
    }
    const zero = (h * maxPos) / (maxPos + maxNeg);
    const sc = h / (maxPos + maxNeg);
    ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, zero); ctx.lineTo(w, zero); ctx.stroke();
    const bw = w / CHART_BARS;
    shown.forEach((g, i) => {
      let up = zero, down = zero;
      const keys = Object.keys(g).sort((a, b) => Number(BONUS_TERMS.has(b)) - Number(BONUS_TERMS.has(a)));
      for (const k of keys) {
        const v = g[k] * sc;
        ctx.fillStyle = TERM_COLORS[k] || "#999";
        if (v >= 0) { ctx.fillRect(i * bw + 1, up - v, bw - 2, v); up -= v; }
        else { ctx.fillRect(i * bw + 1, down, bw - 2, -v); down += -v; }
      }
    });
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.font = `${11 * devicePixelRatio}px system-ui`;
    ctx.fillText(`reward per ${GROUP} frames`, 6, 14 * devicePixelRatio);
  }

  legend(terms) {
    const el = this.$(".legend");
    const keys = new Set(["distance", "speed", "tilt", "legs", "fuel", "terminal", ...terms]);
    el.innerHTML = [...keys].map((k) => `<span><i style="background:${TERM_COLORS[k] || "#999"}"></i>${TERM_LABELS[k] || k}</span>`).join("");
  }
}
