// How hard is each difficulty for a human? A scripted pilot with reaction latency: it sees the
// observation from `delay` frames ago and holds each decision for `hold` frames.
//   node tests/human_sim.mjs [episodes] [delay] [hold]
import { LunarLanderEnv } from "../src/lander_env.js";

const episodes = Number(process.argv[2] ?? 30);
const delay = Number(process.argv[3] ?? 10); // 0.2 s
const hold = Number(process.argv[4] ?? 6);
const MODES = {
  easy: { gravity: -4, assist: 0.9, walls: true },
  medium: { gravity: -6, assist: 0.9, walls: true },
  real: { gravity: -10, assist: 0 },
};

function pilot(s) { // gymnasium's reference heuristic (handles sideways drift)
  const angleTarg = Math.max(-0.4, Math.min(0.4, s[0] * 0.5 + s[2]));
  const hoverTarg = 0.55 * Math.abs(s[0]);
  let angleTodo = (angleTarg - s[4]) * 0.5 - s[5];
  let hoverTodo = (hoverTarg - s[1]) * 0.5 - s[3] * 0.5;
  if (s[6] || s[7]) { angleTodo = 0; hoverTodo = -s[3] * 0.5; }
  if (hoverTodo > Math.abs(angleTodo) && hoverTodo > 0.05) return 2;
  if (angleTodo < -0.05) return 3;
  if (angleTodo > 0.05) return 1;
  return 0;
}
const heuristic = pilot;

for (const [name, cfg] of Object.entries(MODES)) {
  for (const [label, d, h] of [["instant", 0, 1], ["human-like", delay, hold]]) {
    let landed = 0, crashed = 0, timeout = 0;
    for (let ep = 0; ep < episodes; ep++) {
      const env = new LunarLanderEnv(cfg);
      let obs = env.reset(20000 + ep);
      const hist = [obs];
      let a = 0, t = 0, r;
      do {
        if (t % h === 0) a = heuristic(hist[Math.max(0, hist.length - 1 - d)]);
        r = env.step(a);
        hist.push(r.obs);
        t++;
      } while (!env.done);
      if (r.outcome === "landed") landed++; else if (r.outcome === "crashed") crashed++; else timeout++;
    }
    console.log(`${name.padEnd(7)} ${label.padEnd(11)} landed ${(landed / episodes).toFixed(2)} crashed ${(crashed / episodes).toFixed(2)} timeout ${(timeout / episodes).toFixed(2)}`);
  }
}
