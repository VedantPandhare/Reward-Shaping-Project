// App shell: drivers, controls, fixed-timestep loop, results modal.
import { LunarLanderEnv } from "./lander_env.js";
import { act } from "./policy.js";
import { VARIANTS } from "./rewards.js";
import { Panel } from "./panel.js";

const STEP = 1 / 50;
const WINDS = [{ windPower: 0, turbulencePower: 0 }, { windPower: 6, turbulencePower: 0.6 }, { windPower: 15, turbulencePower: 1.5 }];
const CUSTOM_TERMS = {
  time_penalty: ["time_penalty"], hover_hack: ["hover_bonus"], tilt_only_hack: ["tilt_bonus"],
  delta_fix: ["delta_shaping", "time_penalty"], delta_no_time: ["delta_shaping"],
  fuel_efficient: ["extra_fuel"], soft_landing: ["touchdown_penalty"],
};
const css = (n) => `#${n.toString(16).padStart(6, "0")}`;

function driverFor(id) {
  if (id === "human") {
    return { id, human: true, variant: "baseline", label: "You", blurb: "keyboard pilot", color: 0x00e5ff, cssColor: css(0x00e5ff), hasBonus: false, terms: [] };
  }
  const v = VARIANTS[id];
  const color = v.hack ? 0xff5a3c : id.startsWith("delta") ? 0x3ddc97 : id === "baseline" ? 0x4d8dff : 0xb388eb;
  return { id, human: false, variant: id, label: `${v.label} agent`, blurb: v.blurb, color, cssColor: css(color), hasBonus: id in CUSTOM_TERMS, terms: CUSTOM_TERMS[id] || [] };
}

const $ = (s) => document.querySelector(s);
const arena = $("#arena");
const tpl = $("#panel-tpl");
const panels = [0, 1].map(() => {
  const node = tpl.content.firstElementChild.cloneNode(true);
  arena.appendChild(node);
  return new Panel(node);
});

const state = { mode: "idle", left: "delta_fix", right: "hover_hack", seed: 1, wind: 0, speed: 1, acc: 0, countdown: 0, envs: [], drivers: [], done: [false, false] };
let policies = null;

function fillSelect(sel, withHuman, value) {
  sel.innerHTML = (withHuman ? [`<option value="human">You (keyboard)</option>`] : [])
    .concat(Object.entries(VARIANTS).map(([k, v]) => `<option value="${k}">${v.label} agent</option>`)).join("");
  sel.value = value;
}
fillSelect($("#sel-left"), true, state.left);
fillSelect($("#sel-right"), false, state.right);

// ---- human input ----
const keys = { left: false, up: false, right: false };
const KEYMAP = { ArrowLeft: "left", a: "left", A: "left", ArrowUp: "up", w: "up", W: "up", ArrowRight: "right", d: "right", D: "right" };
addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (KEYMAP[e.key]) { keys[KEYMAP[e.key]] = true; e.preventDefault(); }
  if (e.key === " ") { e.preventDefault(); start(); }
});
addEventListener("keyup", (e) => { if (KEYMAP[e.key]) keys[KEYMAP[e.key]] = false; });
document.querySelectorAll("#touch button").forEach((b) => {
  const k = b.dataset.k;
  const on = (v) => (e) => { keys[k] = v; e.preventDefault(); };
  b.addEventListener("pointerdown", on(true)); b.addEventListener("pointerup", on(false)); b.addEventListener("pointerleave", on(false));
});
const humanAction = () => (keys.up ? 2 : keys.left ? 1 : keys.right ? 3 : 0);

// ---- episode control ----
function start() {
  state.drivers = [driverFor(state.left), driverFor(state.right)];
  const cfg = WINDS[state.wind];
  state.envs = state.drivers.map((d) => {
    const env = new LunarLanderEnv({ variant: d.variant, ...cfg });
    env.reset(state.seed);
    return env;
  });
  state.drivers.forEach((d, i) => { panels[i].setup(state.envs[i], d); panels[i].legend(d.terms); });
  state.done = [false, false];
  state.acc = 0;
  state.mode = "countdown";
  state.countdown = state.drivers[0].human ? 3.2 : 1.2;
  $("#resultcard").classList.remove("on");
  const humanOn = state.drivers[0].human;
  $("#sel-speed").disabled = humanOn;
  if (humanOn) { state.speed = 1; $("#sel-speed").value = "1"; }
  $("#btn-play").textContent = humanOn ? "\u{1F916} Watch AI vs AI" : "\u{1F3AE} Take the controls";
  $("#in-seed").value = state.seed;
}

function stepAll() {
  for (let i = 0; i < 2; i++) {
    if (state.done[i]) continue;
    const env = state.envs[i], d = state.drivers[i];
    const a = d.human ? humanAction() : act(policies[d.variant], env.obs);
    const r = env.step(a);
    panels[i].tick(r, a);
    if (env.done) state.done[i] = true;
  }
  if (state.done[0] && state.done[1]) showResult();
}

function outcomeText(i) {
  const env = state.envs[i], d = state.drivers[i];
  const o = env.outcome === "landed" ? "landed" : env.outcome === "crashed" ? "crashed" : "timed out without landing";
  return `<b>${d.label}</b> ${o} (score ${env.returns.raw.toFixed(0)})`;
}

function showResult() {
  state.mode = "finished";
  const [a, b] = state.envs;
  let line = `${outcomeText(0)} &nbsp;vs&nbsp; ${outcomeText(1)}`;
  const hacker = state.drivers.findIndex((d, i) => VARIANTS[d.id]?.hack && state.envs[i].outcome === "timeout");
  if (hacker >= 0) {
    const e = state.envs[hacker];
    line += `<br><small style="display:inline;text-transform:none">The hack agent banked a shaped reward of <b>${e.returns.shaped.toFixed(0)}</b> but an env score of only <b>${e.returns.raw.toFixed(0)}</b>: reward went up, behaviour got worse.</small>`;
  } else if (state.drivers[0].human) {
    const win = a.returns.raw > b.returns.raw && a.outcome === "landed";
    line += `<br><small style="display:inline;text-transform:none">${win ? "You beat the AI. " : "AI wins this round. "}Press Space to go again.</small>`;
  }
  const card = $("#resultcard");
  card.innerHTML = line;
  card.classList.add("on");
}

// ---- main loop ----
let last = performance.now();
function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  const cd = $("#countdown");
  if (state.mode === "countdown") {
    state.countdown -= dt;
    cd.classList.add("on");
    cd.textContent = state.countdown > 0.2 ? String(Math.ceil(state.countdown)) : "GO";
    if (state.countdown <= 0) { cd.classList.remove("on"); state.mode = "running"; }
  } else if (state.mode === "running") {
    state.acc += dt * state.speed;
    let n = 0;
    while (state.acc >= STEP && n++ < 12 && state.mode === "running") { stepAll(); state.acc -= STEP; }
    if (n >= 12) state.acc = 0;
  }
  if (state.envs.length) {
    const alpha = state.mode === "running" ? state.acc / STEP : 1;
    panels.forEach((p) => p.render(dt, now / 1000, alpha));
  }
  requestAnimationFrame(frame);
}

// ---- controls ----
$("#sel-left").onchange = (e) => { state.left = e.target.value; start(); };
$("#sel-right").onchange = (e) => { state.right = e.target.value; start(); };
$("#sel-wind").onchange = (e) => { state.wind = Number(e.target.value); start(); };
$("#sel-speed").onchange = (e) => { state.speed = Number(e.target.value); };
$("#in-seed").onchange = (e) => { state.seed = Math.max(0, Number(e.target.value) | 0); start(); };
$("#btn-new").onclick = () => { state.seed = Math.floor(Math.random() * 100000); start(); };
$("#btn-restart").onclick = () => start();
$("#btn-play").onclick = () => {
  state.left = state.drivers[0]?.human ? "delta_fix" : "human";
  $("#sel-left").value = state.left;
  start();
};
addEventListener("resize", () => panels.forEach((p) => p.resize()));

function openModal(id) { $(id).classList.add("on"); }
document.querySelectorAll(".modal").forEach((m) => {
  m.addEventListener("click", (e) => { if (e.target === m || e.target.classList.contains("close")) m.classList.remove("on"); });
});
$("#btn-about").onclick = () => openModal("#modal-about");
$("#btn-results").onclick = async () => { await buildResults(); openModal("#modal-results"); };

let resultsBuilt = false;
async function buildResults() {
  if (resultsBuilt) return;
  resultsBuilt = true;
  const body = $("#results-body");
  try {
    const rows = await (await fetch("results/ablation.json")).json();
    const pm = (r, k, d) => `${r[k + "_mean"].toFixed(d)} &plusmn; ${r[k + "_std"].toFixed(d)}`;
    const cls = (r) => (VARIANTS[r.variant]?.hack ? "hack" : r.variant === "delta_fix" ? "fix" : "");
    body.innerHTML = `<table><tr><th>variant</th><th>return</th><th>landing</th><th>timeout</th><th>ep. length</th><th>fuel</th><th>hack flag</th></tr>${
      rows.map((r) => `<tr class="${cls(r)}"><td>${r.variant}</td><td>${pm(r, "mean_return", 0)}</td><td>${pm(r, "landing_rate", 2)}</td><td>${pm(r, "timeout_rate", 2)}</td><td>${pm(r, "mean_length", 0)}</td><td>${pm(r, "mean_fuel", 0)}</td><td>${r.hack_flag}</td></tr>`).join("")}</table>
      <p style="color:#8c95c4">fuel_efficient and soft_landing were trained for 500k steps instead of 1M. Scores come from the unmodified environment reward.</p>
      <div class="imgs"><img src="results/learning_curves.png" alt="learning curves"><img src="results/tradeoff.png" alt="trade-off"></div>`;
  } catch (e) {
    body.textContent = "Results not found.";
  }
}

window.__lab = { state, advance(n) { state.mode = "running"; $("#countdown").classList.remove("on"); for (let i = 0; i < n && state.mode === "running"; i++) stepAll(); } };

(async () => {
  policies = await (await fetch("models/policies.json")).json();
  start();
  requestAnimationFrame(frame);
})();
