// JS mirror of rewardlab.wrappers: per-variant extra reward + breakdown of the built-in reward.

export const VARIANTS = {
  baseline: { label: "Baseline", blurb: "Built-in reward only.", hack: false },
  time_penalty: { label: "Time penalty", blurb: "-0.05 every frame.", hack: false },
  hover_hack: {
    label: "Hover hack", hack: true,
    blurb: "+1 per frame while upright and slow. Hovering out-earns landing.",
  },
  delta_fix: {
    label: "Delta fix", hack: false,
    blurb: "Pays for improvement (potential shaping) + small time cost. No loophole.",
  },
  delta_no_time: { label: "Delta, no time penalty", blurb: "Improvement-based only.", hack: false },
  tilt_only_hack: {
    label: "Tilt-only hack", hack: true, blurb: "+1 per frame for low tilt. Same exploit.",
  },
  fuel_efficient: { label: "Fuel efficient", blurb: "Doubled fuel cost.", hack: false },
  soft_landing: { label: "Soft landing", blurb: "Penalty on touchdown speed and tilt.", hack: false },
};

const speed = (o) => Math.hypot(o[2], o[3]);
const phi = (o) => -(Math.abs(o[4]) + speed(o));
const MAIN = 0.3;
const SIDE = 0.03;
const fuelCost = (a) => (a === 2 ? MAIN : a === 1 || a === 3 ? SIDE : 0);

/** Per-episode reward function for a variant. Returns {extra, terms}. */
export function makeVariantReward(name, gamma = 0.999) {
  let prev = 0;
  return {
    reset(obs) { prev = phi(obs); },
    step(obs, action, terminated) {
      switch (name) {
        case "time_penalty":
          return { time_penalty: -0.05 };
        case "hover_hack": {
          const ok = Math.abs(obs[4]) < 0.2 && speed(obs) < 0.3;
          return { hover_bonus: ok ? 1 : 0 };
        }
        case "tilt_only_hack":
          return { tilt_bonus: Math.abs(obs[4]) < 0.2 ? 1 : 0 };
        case "delta_fix":
        case "delta_no_time": {
          const p = terminated ? 0 : phi(obs);
          const shaping = 10 * (gamma * p - prev);
          prev = p;
          const t = { delta_shaping: shaping };
          if (name === "delta_fix") t.time_penalty = -0.05;
          return t;
        }
        case "fuel_efficient":
          return { extra_fuel: -(2 - 1) * fuelCost(action) };
        case "soft_landing":
          return terminated && Math.abs(obs[0]) < 1
            ? { touchdown_penalty: -10 * (Math.abs(obs[3]) + Math.abs(obs[4])) }
            : {};
        default:
          return {};
      }
    },
  };
}

function shapingParts(o) {
  return {
    distance: -100 * Math.hypot(o[0], o[1]),
    speed: -100 * speed(o),
    tilt: -100 * Math.abs(o[4]),
    legs: 10 * o[6] + 10 * o[7],
  };
}

/** Split the built-in per-frame reward into terms (matches rewardlab.wrappers.env_breakdown). */
export function envBreakdown(prevObs, obs, action, reward) {
  const a = shapingParts(prevObs);
  const b = shapingParts(obs);
  const terms = {};
  let sum = 0;
  for (const k of Object.keys(a)) { terms[k] = b[k] - a[k]; sum += terms[k]; }
  terms.fuel = -fuelCost(action);
  sum += terms.fuel;
  terms.terminal = Math.abs(reward - sum) < 1e-3 ? 0 : reward - sum;
  return terms;
}
