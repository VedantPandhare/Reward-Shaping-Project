// Renders the results tables from the exported JSON so the page always matches the evaluation data.
const PLAIN = {
  baseline: "built-in reward only", time_penalty: "&minus;0.05 per frame", hover_hack: "+1 per frame for hovering",
  delta_fix: "improvement-based + time cost", delta_no_time: "improvement-based only", tilt_only_hack: "+1 per frame for staying level",
  fuel_efficient: "fuel cost doubled", soft_landing: "touchdown penalty",
};
const HACKS = new Set(["hover_hack", "tilt_only_hack"]);

const pm = (r, k, d) => `${r[k + "_mean"].toFixed(d)} <span class="sub">&plusmn; ${r[k + "_std"].toFixed(d)}</span>`;

function table(rows, withFuel) {
  const head = `<tr><th>Reward design</th><th class="num">Seeds</th><th class="num">Score</th><th class="num">Landed</th><th class="num">Crashed</th>
    <th class="num">Timed out</th><th class="num">Frames</th>${withFuel ? '<th class="num">Fuel</th>' : ""}<th class="num">Detector flags</th></tr>`;
  const body = rows.map((r) => {
    const cls = HACKS.has(r.variant) ? "bad" : r.variant === "delta_fix" ? "good" : "";
    return `<tr class="${cls}"><td><code>${r.variant}</code><span class="sub">${PLAIN[r.variant] || ""}</span></td>
      <td class="num">${r.n_seeds}</td><td class="num">${pm(r, "mean_return", 0)}</td><td class="num">${pm(r, "landing_rate", 2)}</td>
      <td class="num">${pm(r, "crash_rate", 2)}</td><td class="num">${pm(r, "timeout_rate", 2)}</td><td class="num">${pm(r, "mean_length", 0)}</td>
      ${withFuel ? `<td class="num">${pm(r, "mean_fuel", 0)}</td>` : ""}<td class="num">${r.hack_flag}</td></tr>`;
  }).join("");
  return `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

async function load(file, target, withFuel, note) {
  const el = document.getElementById(target);
  try {
    const rows = await (await fetch(file)).json();
    el.innerHTML = table(rows, withFuel) + (note ? `<p class="note">${note}</p>` : "");
  } catch {
    el.textContent = "Table unavailable (serve this folder over HTTP, e.g. python -m http.server).";
  }
}

load("results/ablation.json", "ablation-table", true,
  "Score is the raw environment reward (a safe landing is about +250 to +270). Detector flags: runs flagged out of seeds. Rates are rounded: <code>hover_hack</code> and <code>tilt_only_hack</code> each timed out in 299 of 300 evaluation episodes. <code>fuel_efficient</code> and <code>soft_landing</code> were trained for 500k steps.");
load("results/ablation_dqn.json", "dqn-table", false, "DQN, 500k steps, 3 seeds.");

// lazy-start gallery videos when they scroll into view
const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    const v = e.target;
    if (e.isIntersecting) {
      if (v.dataset.src && !v.src) v.src = v.dataset.src;
      v.play?.().catch(() => {});
    } else v.pause?.();
  }
}, { rootMargin: "200px" });
document.querySelectorAll(".vidgrid video").forEach((v) => io.observe(v));
