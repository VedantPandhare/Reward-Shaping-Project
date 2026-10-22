// Three.js scene pieces: starfield, earth, terrain, helipad, lander model, particles.
import * as THREE from "three";
import { W, H } from "./lander_env.js";

const rnd = (a, b) => a + Math.random() * (b - a);

function noiseTexture(size = 256, base = [138, 141, 150], amp = 38) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d");
  const img = g.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const n = (Math.random() - 0.5) * amp + (Math.sin(i * 0.013) * 6);
    img.data[i * 4] = base[0] + n; img.data[i * 4 + 1] = base[1] + n; img.data[i * 4 + 2] = base[2] + n; img.data[i * 4 + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  // craters
  for (let i = 0; i < 26; i++) {
    const x = rnd(0, size), y = rnd(0, size), r = rnd(5, 22);
    const grad = g.createRadialGradient(x, y, r * 0.2, x, y, r);
    grad.addColorStop(0, "rgba(40,42,50,0.55)");
    grad.addColorStop(0.8, "rgba(40,42,50,0.25)");
    grad.addColorStop(1, "rgba(200,205,215,0.35)");
    g.fillStyle = grad;
    g.beginPath(); g.arc(x, y, r, 0, Math.PI * 2); g.fill();
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function earthTexture() {
  const w = 512, h = 256;
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const g = c.getContext("2d");
  g.fillStyle = "#1b4f9c"; g.fillRect(0, 0, w, h);
  for (let i = 0; i < 70; i++) {
    g.fillStyle = Math.random() < 0.7 ? "#3f8a3c" : "#a98f58";
    g.beginPath(); g.ellipse(rnd(0, w), rnd(30, h - 30), rnd(14, 55), rnd(8, 30), rnd(0, 3), 0, Math.PI * 2); g.fill();
  }
  g.fillStyle = "#eef3f8"; g.fillRect(0, 0, w, 14); g.fillRect(0, h - 14, w, 14);
  for (let i = 0; i < 90; i++) {
    g.fillStyle = `rgba(255,255,255,${rnd(0.15, 0.45)})`;
    g.beginPath(); g.ellipse(rnd(0, w), rnd(0, h), rnd(20, 60), rnd(4, 11), rnd(-0.3, 0.3), 0, Math.PI * 2); g.fill();
  }
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export function buildSky(scene) {
  const n = 1800;
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    pos[i * 3] = rnd(-160, 170); pos[i * 3 + 1] = rnd(-10, 110); pos[i * 3 + 2] = rnd(-170, -110);
    const b = rnd(0.35, 1);
    col[i * 3] = b; col[i * 3 + 1] = b * rnd(0.85, 1); col[i * 3 + 2] = b * rnd(0.9, 1.1);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const stars = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.55, vertexColors: true, sizeAttenuation: true, fog: false }));
  scene.add(stars);

  const earth = new THREE.Mesh(
    new THREE.SphereGeometry(7, 48, 32),
    new THREE.MeshStandardMaterial({ map: earthTexture(), roughness: 0.9, emissive: 0x0a1830, emissiveIntensity: 0.6 }));
  earth.position.set(-12, 34, -95);
  scene.add(earth);
  const glow = new THREE.Mesh(
    new THREE.SphereGeometry(7.5, 32, 24),
    new THREE.MeshBasicMaterial({ color: 0x5aa0ff, transparent: true, opacity: 0.12, blending: THREE.AdditiveBlending, side: THREE.BackSide, depthWrite: false }));
  glow.position.copy(earth.position);
  scene.add(glow);
  return { earth };
}

function ridge(color, zDepth, base, amp, seed) {
  const shape = new THREE.Shape();
  shape.moveTo(-120, -8);
  let x = -120, i = 0;
  while (x <= 140) {
    const y = base + amp * (Math.sin(i * 0.7 + seed) * 0.5 + Math.sin(i * 1.9 + seed * 2) * 0.35 + Math.random() * 0.25);
    shape.lineTo(x, y); x += rnd(3, 7); i++;
  }
  shape.lineTo(140, -8);
  const m = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshStandardMaterial({ color, roughness: 1, flatShading: true }));
  m.position.z = zDepth;
  return m;
}

/** Terrain from env.terrain (list of [x, y]); returns {group, heightAt}. */
export function buildTerrain(terrain, helipad) {
  const group = new THREE.Group();
  const first = terrain[0], last = terrain[terrain.length - 1];
  const pts = [[-60, first[1]], ...terrain, [W + 60, last[1]]];
  const shape = new THREE.Shape();
  shape.moveTo(pts[0][0], -6);
  for (const [x, y] of pts) shape.lineTo(x, y);
  shape.lineTo(pts[pts.length - 1][0], -6);
  const depth = 34;
  const geo = new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: true, bevelSize: 0.06, bevelThickness: 0.06, bevelSegments: 1 });
  geo.translate(0, 0, -depth + 1.1);
  const tex = noiseTexture();
  tex.repeat.set(0.35, 0.35);
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95, metalness: 0.02, color: 0xffffff }));
  mesh.receiveShadow = true; mesh.castShadow = true;
  group.add(mesh);

  const heightAt = (x) => {
    for (let i = 0; i < pts.length - 1; i++) {
      const [x0, y0] = pts[i], [x1, y1] = pts[i + 1];
      if (x >= x0 && x <= x1) return y0 + ((y1 - y0) * (x - x0)) / (x1 - x0);
    }
    return 0;
  };

  // rocks
  const rockGeo = new THREE.IcosahedronGeometry(1, 0);
  const rockMat = new THREE.MeshStandardMaterial({ color: 0x70747d, roughness: 1, flatShading: true });
  const rocks = new THREE.InstancedMesh(rockGeo, rockMat, 140);
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), e = new THREE.Euler(), s = new THREE.Vector3(), p = new THREE.Vector3();
  for (let i = 0; i < 140; i++) {
    let x = rnd(-14, W + 14);
    if (x > helipad.x1 - 1.2 && x < helipad.x2 + 1.2) x += x < W / 2 ? -4 : 4;
    const z = rnd(-26, 0.9);
    const sc = rnd(0.05, 0.32) * (z < -4 ? 1.8 : 1);
    p.set(x, heightAt(x) + sc * 0.2, z);
    e.set(rnd(0, 6), rnd(0, 6), rnd(0, 6));
    q.setFromEuler(e); s.set(sc * rnd(0.8, 1.5), sc, sc * rnd(0.8, 1.5));
    m4.compose(p, q, s);
    rocks.setMatrixAt(i, m4);
  }
  rocks.castShadow = true; rocks.receiveShadow = true;
  group.add(rocks);

  group.add(ridge(0x2a2c35, -42, 4, 7, 1.3));
  group.add(ridge(0x1d1f27, -62, 8, 11, 4.1));

  // helipad: raised slab with beacon lights and flags
  const padW = helipad.x2 - helipad.x1;
  const pad = new THREE.Mesh(new THREE.BoxGeometry(padW, 0.12, 3.2),
    new THREE.MeshStandardMaterial({ color: 0x23272e, roughness: 0.6, metalness: 0.4 }));
  pad.position.set((helipad.x1 + helipad.x2) / 2, helipad.y - 0.06, 0.5);
  pad.receiveShadow = true;
  group.add(pad);
  const stripe = new THREE.Mesh(new THREE.PlaneGeometry(padW * 0.9, 0.08),
    new THREE.MeshBasicMaterial({ color: 0xffffff }));
  stripe.rotation.x = -Math.PI / 2;
  stripe.position.set(pad.position.x, helipad.y + 0.005, 0.5);
  group.add(stripe);
  const beaconMat = new THREE.MeshBasicMaterial({ color: 0x40ffb0 });
  const beacons = [];
  for (const x of [helipad.x1, helipad.x2]) {
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 1.6, 8), new THREE.MeshStandardMaterial({ color: 0xdddddd }));
    pole.position.set(x, helipad.y + 0.8, 0.5);
    group.add(pole);
    const flag = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.33), new THREE.MeshBasicMaterial({ color: 0xffd400, side: THREE.DoubleSide }));
    flag.position.set(x + (x === helipad.x1 ? 0.4 : -0.4), helipad.y + 1.4, 0.5);
    group.add(flag);
    const light = new THREE.Mesh(new THREE.SphereGeometry(0.09, 12, 8), beaconMat.clone());
    light.position.set(x, helipad.y + 1.65, 0.5);
    group.add(light);
    beacons.push(light);
  }
  for (let i = 0; i < 7; i++) { // pad edge lights
    const l = new THREE.Mesh(new THREE.SphereGeometry(0.05, 8, 6), new THREE.MeshBasicMaterial({ color: 0xff5a3c }));
    l.position.set(helipad.x1 + (padW * i) / 6, helipad.y + 0.04, 2.05);
    group.add(l);
    beacons.push(l);
  }
  return { group, heightAt, beacons };
}

// ---------- particles ----------
export class Particles {
  constructor(max = 420) {
    this.max = max;
    this.pos = new Float32Array(max * 3);
    this.col = new Float32Array(max * 3);
    this.size = new Float32Array(max);
    this.alpha = new Float32Array(max);
    this.vel = new Float32Array(max * 3);
    this.life = new Float32Array(max);
    this.maxLife = new Float32Array(max).fill(1);
    this.baseSize = new Float32Array(max);
    this.baseCol = new Float32Array(max * 3);
    this.drag = new Float32Array(max);
    this.next = 0;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(this.pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(this.col, 3));
    g.setAttribute("aSize", new THREE.BufferAttribute(this.size, 1));
    g.setAttribute("aAlpha", new THREE.BufferAttribute(this.alpha, 1));
    this.uniforms = { uScale: { value: 400 } };
    this.points = new THREE.Points(g, new THREE.ShaderMaterial({
      uniforms: this.uniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      vertexShader: `attribute float aSize; attribute float aAlpha; varying vec3 vCol; varying float vA; uniform float uScale;
        void main(){ vCol=color; vA=aAlpha; vec4 mv=modelViewMatrix*vec4(position,1.); gl_PointSize=aSize*uScale/-mv.z; gl_Position=projectionMatrix*mv; }`,
      fragmentShader: `varying vec3 vCol; varying float vA;
        void main(){ float d=length(gl_PointCoord-0.5)*2.; if(d>1.) discard; float f=smoothstep(1.,0.,d); gl_FragColor=vec4(vCol*f*vA, f*vA); }`,
      vertexColors: true,
    }));
    this.points.frustumCulled = false;
  }

  emit(x, y, z, vx, vy, vz, life, size, rgb, drag = 1.5) {
    const i = this.next; this.next = (this.next + 1) % this.max;
    this.pos.set([x, y, z], i * 3); this.vel.set([vx, vy, vz], i * 3);
    this.life[i] = this.maxLife[i] = life; this.baseSize[i] = size; this.baseCol.set(rgb, i * 3); this.drag[i] = drag;
  }

  update(dt, gravity = -1.6) {
    for (let i = 0; i < this.max; i++) {
      if (this.life[i] <= 0) { this.alpha[i] = 0; continue; }
      this.life[i] -= dt;
      const k = Math.max(this.life[i], 0) / this.maxLife[i];
      const d = Math.exp(-this.drag[i] * dt);
      this.vel[i * 3] *= d; this.vel[i * 3 + 1] = this.vel[i * 3 + 1] * d + gravity * dt; this.vel[i * 3 + 2] *= d;
      for (let a = 0; a < 3; a++) this.pos[i * 3 + a] += this.vel[i * 3 + a] * dt;
      this.alpha[i] = k;
      this.size[i] = this.baseSize[i] * (0.6 + 0.8 * (1 - k));
      for (let a = 0; a < 3; a++) this.col[i * 3 + a] = this.baseCol[i * 3 + a];
    }
    const g = this.points.geometry;
    for (const n of ["position", "color", "aSize", "aAlpha"]) g.attributes[n].needsUpdate = true;
  }
}

// ---------- lander model ----------
const LANDER_POLY = [[-14, 17], [-17, 0], [-17, -10], [17, -10], [17, 0], [14, 17]];
const flameMat = (c, o) => new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: o, blending: THREE.AdditiveBlending, depthWrite: false });

function strut(mat) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 1, 6), mat);
  m.castShadow = true;
  return m;
}

function placeStrut(mesh, a, b) {
  const dx = b.x - a.x, dy = b.y - a.y;
  mesh.position.set((a.x + b.x) / 2, (a.y + b.y) / 2, a.z);
  mesh.scale.set(1, Math.hypot(dx, dy), 1);
  mesh.rotation.z = Math.atan2(dy, dx) - Math.PI / 2;
}

export function buildLander(color) {
  const root = new THREE.Group();
  const body = new THREE.Group();
  root.add(body);
  const shape = new THREE.Shape(LANDER_POLY.map(([x, y]) => new THREE.Vector2(x / 30, y / 30)));
  const hullGeo = new THREE.ExtrudeGeometry(shape, { depth: 0.7, bevelEnabled: true, bevelSize: 0.05, bevelThickness: 0.05, bevelSegments: 2 });
  hullGeo.translate(0, 0, -0.35);
  const hullMat = new THREE.MeshStandardMaterial({ color, metalness: 0.65, roughness: 0.32 });
  const hull = new THREE.Mesh(hullGeo, hullMat);
  hull.castShadow = true; hull.receiveShadow = true;
  body.add(hull);
  const foil = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.18, 0.78), new THREE.MeshStandardMaterial({ color: 0xd9a441, metalness: 0.9, roughness: 0.25 }));
  foil.position.y = -0.2; foil.castShadow = true;
  body.add(foil);
  const win = new THREE.Mesh(new THREE.SphereGeometry(0.2, 20, 14), new THREE.MeshStandardMaterial({ color: 0x9fe8ff, emissive: 0x3aa8e0, emissiveIntensity: 1.4, roughness: 0.1 }));
  win.scale.z = 0.7; win.position.set(0, 0.25, 0.4);
  body.add(win);
  const ant = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.4, 6), new THREE.MeshStandardMaterial({ color: 0xcccccc }));
  ant.position.set(0.25, 0.75, 0);
  body.add(ant);
  const antTip = new THREE.Mesh(new THREE.SphereGeometry(0.04, 8, 6), new THREE.MeshBasicMaterial({ color: 0xff3030 }));
  antTip.position.set(0.25, 0.97, 0);
  body.add(antTip);
  const nozzle = new THREE.Mesh(new THREE.ConeGeometry(0.2, 0.3, 16, 1, true), new THREE.MeshStandardMaterial({ color: 0x2c2f36, metalness: 0.8, roughness: 0.4, side: THREE.DoubleSide }));
  nozzle.rotation.x = Math.PI; nozzle.position.set(0, -0.5, 0);
  body.add(nozzle);
  for (const sx of [-1, 1]) {
    const t = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.14, 0.3), new THREE.MeshStandardMaterial({ color: 0x3a3e47, metalness: 0.7, roughness: 0.4 }));
    t.position.set(sx * 0.6, 0.2, 0);
    body.add(t);
  }

  // flames (local to body)
  const mainFlame = new THREE.Group();
  const cone = (r, h, mat) => {
    const g = new THREE.ConeGeometry(r, h, 14, 1, true);
    g.translate(0, h / 2, 0); // base at origin, tip at +Y
    return new THREE.Mesh(g, mat);
  };
  const outer = cone(0.2, 1.3, flameMat(0xff7a1a, 0.75));
  const inner = cone(0.1, 0.8, flameMat(0xfff2c0, 0.95));
  mainFlame.add(outer, inner);
  mainFlame.rotation.z = Math.PI; // tip points down
  mainFlame.position.set(0, -0.62, 0);
  mainFlame.visible = false;
  body.add(mainFlame);

  const sideFlames = [];
  for (const dir of [-1, 1]) {
    const f = cone(0.08, 0.5, flameMat(0xffa040, 0.85));
    f.rotation.z = -dir * Math.PI / 2; // tip points outward
    f.position.set(dir * 0.7, 0.2, 0);
    f.visible = false;
    body.add(f);
    sideFlames.push(f);
  }
  const light = new THREE.PointLight(0xff9a40, 0, 9, 2);
  light.position.set(0, -1.0, 1.2);
  body.add(light);

  // legs follow physics bodies
  const legMat = new THREE.MeshStandardMaterial({ color: 0xc9ccd4, metalness: 0.7, roughness: 0.35 });
  const legs = [0, 1].map(() => {
    const g = new THREE.Group();
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.133, 0.533, 0.16), legMat);
    bar.castShadow = true;
    const foot = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.2, 0.05, 14), legMat);
    foot.position.y = -0.28; foot.castShadow = true;
    g.add(bar, foot);
    root.add(g);
    return g;
  });
  const struts = [strut(legMat), strut(legMat)];
  struts.forEach((s) => root.add(s));
  const tint = (c) => { hullMat.color.set(c); };
  return { root, body, hullMat, mainFlame, outer, inner, sideFlames, light, legs, struts, tint, placeStrut };
}
