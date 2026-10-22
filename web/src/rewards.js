// JS mirror of rewardlab.wrappers: per-variant extra reward + breakdown of the built-in reward.

// Plain-English descriptions: `rule` says what the agent was paid for during training.
export const VARIANTS = {
  baseline: {
    label: "Standard agent", short: "Standard agent (built-in reward)", hack: false, badge: "Reference",
    blurb: "Trained on the game's own reward.",
    rule: "Paid only by the game's built-in reward: points for getting closer, slower and more upright, a small fuel cost, +100 for landing and -100 for crashing.",
  },
  time_penalty: {
    label: "Hurry agent", short: "Hurry agent (time cost)", hack: false, badge: "Variant",
    blurb: "Built-in reward plus a cost for every frame.",
    rule: "Built-in reward, plus -0.05 points every frame, so wasting time costs points.",
  },
  hover_hack: {
    label: "Hover-hack agent", short: "Hover-hack agent (rewarded for hovering)", hack: true, badge: "Exploits its reward",
    blurb: "Paid +1 per frame for hovering.",
    rule: "Built-in reward, plus +1 point every frame it is upright and slow. Landing ends the game and stops those points, so it learned to hover until time runs out.",
  },
  delta_fix: {
    label: "Improvement-reward agent", short: "Improvement-reward agent (the fix)", hack: false, badge: "The fix",
    blurb: "Paid only for getting better.",
    rule: "Built-in reward, plus a bonus only when it gets steadier than the frame before (the change in an upright-and-slow score), and -0.05 per frame. Hovering earns nothing, so landing wins.",
  },
  delta_no_time: {
    label: "Improvement-only agent", short: "Improvement-only agent (no time cost)", hack: false, badge: "Variant",
    blurb: "Improvement bonus without the time cost.",
    rule: "Same improvement bonus as the fix, but no per-frame cost. Tests whether the time cost was needed (it was not).",
  },
  tilt_only_hack: {
    label: "Stay-level-hack agent", short: "Stay-level-hack agent (second exploit)", hack: true, badge: "Exploits its reward",
    blurb: "Paid +1 per frame for staying level.",
    rule: "Built-in reward, plus +1 point every frame it is level, ignoring where it is or how fast it moves. Another loophole: stay level, never land.",
  },
  fuel_efficient: {
    label: "Fuel-saver agent", short: "Fuel-saver agent (fuel costs double)", hack: false, badge: "Trade-off",
    blurb: "Fuel costs twice as much.",
    rule: "Built-in reward, but every engine burn costs double. A trade-off, not a hack: it burns less fuel and lands less reliably. Trained for half as long (500k steps).",
  },
  soft_landing: {
    label: "Gentle-landing agent", short: "Gentle-landing agent (touchdown penalty)", hack: false, badge: "Trade-off",
    blurb: "Penalised for hard or tilted touchdowns.",
    rule: "Built-in reward, plus a penalty for touchdown speed and tilt. A trade-off, not a hack. Trained for half as long (500k steps).",
  },
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
