// LunarLander-v3 physics port (planck.js = Box2D). Mirrors gymnasium/envs/box2d/lunar_lander.py.
import * as planck from "../vendor/planck.mjs";
import { envBreakdown, makeVariantReward } from "./rewards.js";

const { Vec2 } = planck;
export const FPS = 50;
export const SCALE = 30.0;
const MAIN_ENGINE_POWER = 13.0;
const SIDE_ENGINE_POWER = 0.6;
const INITIAL_RANDOM = 1000.0;
const LANDER_POLY = [[-14, 17], [-17, 0], [-17, -10], [17, -10], [17, 0], [14, 17]];
const LEG_AWAY = 20, LEG_DOWN = 18, LEG_W = 2, LEG_H = 8, LEG_SPRING_TORQUE = 40;
const SIDE_ENGINE_HEIGHT = 14, SIDE_ENGINE_AWAY = 12, MAIN_ENGINE_Y_LOCATION = 4;
export const VIEWPORT_W = 600, VIEWPORT_H = 400;
export const W = VIEWPORT_W / SCALE, H = VIEWPORT_H / SCALE;
export const MAX_STEPS = 1000;
const CHUNKS = 11;

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class LunarLanderEnv {
  /**
   * @param {object} o  variant: reward variant name; windPower/turbulencePower: 0 = off;
   *                    gravity: default -10
   */
  constructor({ variant = "baseline", windPower = 0, turbulencePower = 0, gravity = -10, assist = 0, walls = false } = {}) {
    this.walls = walls; // human-only handicap: bouncy invisible side walls (agents never use them)
    this.assist = assist; // 0 = off; else per-frame angular-velocity retention (e.g. 0.9) when no side engine fires
    this.variant = variant;
    this.windPower = windPower;
    this.turbulencePower = turbulencePower;
    this.gravity = gravity;
    this.windOn = windPower > 0 || turbulencePower > 0;
  }

  /** Same terrain + initial push for the same seed, so two envs are directly comparable. */
  reset(seed = 0) {
    const rng = mulberry32(seed);
    this.uniform = (lo, hi) => lo + (hi - lo) * rng();
    this.rngStep = mulberry32(seed ^ 0x9e3779b9);
    this.world = new planck.World({ gravity: Vec2(0, this.gravity) });
    this.gameOver = false;
    this.prevShaping = null;
    this.t = 0;
    this.done = false;
    this.legContact = [false, false];

    const heights = [];
    for (let i = 0; i < CHUNKS + 1; i++) heights.push(this.uniform(0, H / 2));
    const chunkX = [];
    for (let i = 0; i < CHUNKS; i++) chunkX.push((W / (CHUNKS - 1)) * i);
    this.helipadX1 = chunkX[Math.floor(CHUNKS / 2) - 1];
    this.helipadX2 = chunkX[Math.floor(CHUNKS / 2) + 1];
    this.helipadY = H / 4;
    for (let d = -2; d <= 2; d++) heights[Math.floor(CHUNKS / 2) + d] = this.helipadY;
    const smoothY = [];
    for (let i = 0; i < CHUNKS; i++) {
      const prev = heights[(i - 1 + heights.length) % heights.length]; // python negative index wraps
      smoothY.push(0.33 * (prev + heights[i] + heights[i + 1]));
    }
    this.terrain = chunkX.map((x, i) => [x, smoothY[i]]);

    this.moon = this.world.createBody({ type: "static" });
    this.moon.createFixture(planck.Edge(Vec2(0, 0), Vec2(W, 0)));
    for (let i = 0; i < CHUNKS - 1; i++) {
      this.moon.createFixture(planck.Edge(Vec2(...this.terrain[i]), Vec2(...this.terrain[i + 1])),
        { density: 0, friction: 0.1 });
    }

    if (this.walls) {
      this.wallBody = this.world.createBody({ type: "static" });
      for (const x of [0.05, W - 0.05]) {
        this.wallBody.createFixture(planck.Edge(Vec2(x, 0), Vec2(x, H * 3)), { restitution: 0.6, friction: 0 });
      }
    }

    const initialX = VIEWPORT_W / SCALE / 2;
    const initialY = VIEWPORT_H / SCALE;
    this.lander = this.world.createDynamicBody({ position: Vec2(initialX, initialY), angle: 0 });
    this.lander.createFixture(
      planck.Polygon(LANDER_POLY.map(([x, y]) => Vec2(x / SCALE, y / SCALE))),
      { density: 5.0, friction: 0.1, restitution: 0.0, filterCategoryBits: 0x0010, filterMaskBits: 0x001 });
    this.lander.applyForceToCenter(
      Vec2(this.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), this.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)),
      true);

    if (this.windOn) {
      this.windIdx = Math.floor(this.uniform(-9999, 9999));
      this.torqueIdx = Math.floor(this.uniform(-9999, 9999));
    }

    this.legs = [];
    for (const i of [-1, 1]) {
      const leg = this.world.createDynamicBody({
        position: Vec2(initialX - (i * LEG_AWAY) / SCALE, initialY), angle: i * 0.05,
      });
      leg.createFixture(planck.Box(LEG_W / SCALE, LEG_H / SCALE),
        { density: 1.0, restitution: 0.0, filterCategoryBits: 0x0020, filterMaskBits: 0x001 });
      leg.legIndex = this.legs.length;
      const def = {
        localAnchorA: Vec2(0, 0), localAnchorB: Vec2((i * LEG_AWAY) / SCALE, LEG_DOWN / SCALE),
        enableMotor: true, enableLimit: true, maxMotorTorque: LEG_SPRING_TORQUE, motorSpeed: 0.3 * i,
        lowerAngle: i === -1 ? 0.9 - 0.5 : -0.9, upperAngle: i === -1 ? 0.9 : -0.9 + 0.5,
      };
      this.world.createJoint(planck.RevoluteJoint(def, this.lander, leg));
      this.legs.push(leg);
    }

    this.world.on("begin-contact", (c) => {
      const a = c.getFixtureA().getBody();
      const b = c.getFixtureB().getBody();
      const wall = a === this.wallBody || b === this.wallBody;
      if (!wall && (a === this.lander || b === this.lander)) this.gameOver = true;
      for (const leg of this.legs) if (a === leg || b === leg) this.legContact[leg.legIndex] = true;
    });
    this.world.on("end-contact", (c) => {
      const a = c.getFixtureA().getBody();
      const b = c.getFixtureB().getBody();
      for (const leg of this.legs) if (a === leg || b === leg) this.legContact[leg.legIndex] = false;
    });

    // like gym: reset() performs one no-op step; its reward is discarded
    const first = this._physicsStep(0);
    this.obs = first.obs;
    this.variantReward = makeVariantReward(this.variant);
    this.variantReward.reset(this.obs);
    this.returns = { raw: 0, shaped: 0, fuel: 0, custom: 0, absTotal: 0 };
    return this.obs;
  }

  _physicsStep(action) {
    const lander = this.lander;
    if (this.windOn && !(this.legContact[0] || this.legContact[1])) {
      const windMag = Math.tanh(Math.sin(0.02 * this.windIdx) + Math.sin(Math.PI * 0.01 * this.windIdx)) * this.windPower;
      this.windIdx += 1;
      lander.applyForceToCenter(Vec2(windMag, 0), true);
      const torqueMag = Math.tanh(Math.sin(0.02 * this.torqueIdx) + Math.sin(Math.PI * 0.01 * this.torqueIdx)) * this.turbulencePower;
      this.torqueIdx += 1;
      lander.applyTorque(torqueMag, true);
    }

    const ang = lander.getAngle();
    const tip = [Math.sin(ang), Math.cos(ang)];
    const side = [-tip[1], tip[0]];
    const disp = [(this.rngStep() * 2 - 1) / SCALE, (this.rngStep() * 2 - 1) / SCALE];
    const pos = lander.getPosition();

    let mPower = 0;
    this.flame = { main: false, left: false, right: false, mainPos: null, sidePos: null };
    if (action === 2) {
      mPower = 1.0;
      const ox = tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * disp[0]) + side[0] * disp[1];
      const oy = -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * disp[0]) - side[1] * disp[1];
      const impulsePos = Vec2(pos.x + ox, pos.y + oy);
      lander.applyLinearImpulse(Vec2(-ox * MAIN_ENGINE_POWER * mPower, -oy * MAIN_ENGINE_POWER * mPower), impulsePos, true);
      this.flame.main = true;
    }

    let sPower = 0;
    if (action === 1 || action === 3) {
      const direction = action - 2;
      sPower = 1.0;
      const ox = tip[0] * disp[0] + side[0] * (3 * disp[1] + (direction * SIDE_ENGINE_AWAY) / SCALE);
      const oy = -tip[1] * disp[0] - side[1] * (3 * disp[1] + (direction * SIDE_ENGINE_AWAY) / SCALE);
      const impulsePos = Vec2(pos.x + ox - (tip[0] * 17) / SCALE, pos.y + oy + (tip[1] * SIDE_ENGINE_HEIGHT) / SCALE);
      lander.applyLinearImpulse(Vec2(-ox * SIDE_ENGINE_POWER * sPower, -oy * SIDE_ENGINE_POWER * sPower), impulsePos, true);
      if (direction < 0) this.flame.left = true; else this.flame.right = true;
    }

    if (this.assist && action !== 1 && action !== 3 && !(this.legContact[0] || this.legContact[1])) {
      const w = lander.getAngularVelocity();
      if (Math.abs(w) > 0.02) lander.setAngularVelocity(w * this.assist); // never touch a resting body: it must be able to sleep
    }
    this.world.step(1 / FPS, 6 * 30, 2 * 30);

    const p = lander.getPosition();
    const v = lander.getLinearVelocity();
    const state = [
      (p.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
      (p.y - (this.helipadY + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
      (v.x * (VIEWPORT_W / SCALE / 2)) / FPS,
      (v.y * (VIEWPORT_H / SCALE / 2)) / FPS,
      lander.getAngle(),
      (20.0 * lander.getAngularVelocity()) / FPS,
      this.legContact[0] ? 1 : 0,
      this.legContact[1] ? 1 : 0,
    ];

    let reward = 0;
    const shaping = -100 * Math.hypot(state[0], state[1]) - 100 * Math.hypot(state[2], state[3])
      - 100 * Math.abs(state[4]) + 10 * state[6] + 10 * state[7];
    if (this.prevShaping !== null) reward = shaping - this.prevShaping;
    this.prevShaping = shaping;
    reward -= mPower * 0.3;
    reward -= sPower * 0.03;

    let terminated = false;
    if (this.gameOver || Math.abs(state[0]) >= 1.0) { terminated = true; reward = -100; }
    if (!lander.isAwake()) { terminated = true; reward = 100; }
    return { obs: state, reward, terminated };
  }

  /** Returns {obs, reward(raw), shaped, terminated, truncated, terms, custom, outcome}. */
  step(action) {
    const prev = this.obs;
    const { obs, reward, terminated } = this._physicsStep(action);
    this.t += 1;
    const truncated = !terminated && this.t >= MAX_STEPS;
    const customTerms = this.variantReward.step(obs, action, terminated);
    const custom = Object.values(customTerms).reduce((s, x) => s + x, 0);
    const shaped = reward + custom;
    const terms = { ...envBreakdown(prev, obs, action, reward), ...customTerms };
    this.obs = obs;
    this.done = terminated || truncated;
    const r = this.returns;
    r.raw += reward; r.shaped += shaped; r.custom += custom; r.absTotal += Math.abs(shaped);
    r.fuel += action === 2 ? 0.3 : action === 1 || action === 3 ? 0.03 : 0;
    let outcome = null;
    if (this.done) outcome = truncated ? "timeout" : reward >= 99.9 ? "landed" : "crashed";
    this.outcome = outcome;
    return { obs, reward, shaped, terminated, truncated, terms, custom, outcome };
  }
}
