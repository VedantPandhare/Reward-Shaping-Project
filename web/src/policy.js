// Forward pass of an SB3 PPO MlpPolicy (8 -> 64 tanh -> 64 tanh -> 4), deterministic = argmax.

function linear(w, b, x) {
  const out = new Array(w.length);
  for (let i = 0; i < w.length; i++) {
    let s = b[i];
    const row = w[i];
    for (let j = 0; j < row.length; j++) s += row[j] * x[j];
    out[i] = s;
  }
  return out;
}

export function logits(p, obs) {
  const h1 = linear(p.w1, p.b1, obs).map(Math.tanh);
  const h2 = linear(p.w2, p.b2, h1).map(Math.tanh);
  return linear(p.w3, p.b3, h2);
}

export function act(p, obs) {
  const l = logits(p, obs);
  let best = 0;
  for (let i = 1; i < l.length; i++) if (l[i] > l[best]) best = i;
  return best;
}
