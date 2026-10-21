"""Gradio demo: variant picker, hacked-vs-fixed side by side, live reward breakdown, results tab.

Uses pre-rendered GIFs + breakdown JSON produced by scripts/make_gifs.py (reliable on free CPU
tiers). Run locally: python app/app.py
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
RESULTS = ASSETS if (ASSETS / "ablation.md").exists() else HERE.parent / "results"
GIFS = ASSETS / "gifs" if (ASSETS / "gifs").exists() else HERE.parent / "results" / "gifs"

LABELS = {"baseline": "Baseline (built-in reward)", "hover_hack": "Hover hack (naive bonus)",
          "delta_fix": "Delta fix (improvement-based)", "time_penalty": "Time penalty",
          "delta_no_time": "Delta, no time penalty", "tilt_only_hack": "Tilt-only hack",
          "fuel_efficient": "Fuel efficient", "soft_landing": "Soft landing"}
ENV_TERMS = ["distance", "speed", "tilt", "legs", "fuel", "terminal"]


def available():
    return [v for v in LABELS if (GIFS / f"{v}.gif").exists()]


def load(v):
    return json.loads((GIFS / f"{v}.json").read_text())


def stats(v):
    s = load(v)["summary"]
    outcome = "landed" if s["landed"] else "crashed" if s["crashed"] else "timed out (hovering?)"
    return (f"**{LABELS[v]}** - {outcome}  \nraw return {s['return']:.0f} | "
            f"{s['length']} frames | fuel {s['fuel']:.1f}")


def breakdown_fig(v, window=10):
    steps = load(v)["steps"]
    keys = ENV_TERMS + sorted({k for s in steps for k in s} - set(ENV_TERMS)
                              - {"t", "action", "raw_reward", "total_reward"})
    # bin into windows so the chart is readable; stacked bars of summed reward per window
    n = len(steps) // window + 1
    fig, ax = plt.subplots(figsize=(7, 3.5))
    pos, neg = np.zeros(n), np.zeros(n)
    for k in keys:
        vals = np.array([sum(s.get(k, 0.0) for s in steps[i * window:(i + 1) * window])
                         for i in range(n)])
        if not np.any(vals):
            continue
        bottom = np.where(vals >= 0, pos, neg)
        ax.bar(np.arange(n) * window, vals, width=window, bottom=bottom, label=k)
        pos += np.clip(vals, 0, None)
        neg += np.clip(vals, None, 0)
    ax.set(xlabel="frame", ylabel=f"reward per {window} frames", title=f"Reward breakdown: {v}")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    return fig


def single(v):
    return str(GIFS / f"{v}.gif"), stats(v), breakdown_fig(v)


def side(a, b):
    return (str(GIFS / f"{a}.gif"), stats(a), breakdown_fig(a),
            str(GIFS / f"{b}.gif"), stats(b), breakdown_fig(b))


def build():
    vs = available()
    with gr.Blocks(title="Lunar Lander reward-hacking lab") as demo:
        gr.Markdown("# Lunar Lander reward-hacking lab\nA naive per-frame bonus teaches PPO to "
                    "hover until timeout. An improvement-based reward fixes it.")
        with gr.Tab("Side by side"):
            with gr.Row():
                a = gr.Dropdown(vs, value="hover_hack" if "hover_hack" in vs else vs[0],
                                label="Left")
                b = gr.Dropdown(vs, value="delta_fix" if "delta_fix" in vs else vs[0],
                                label="Right")
            go = gr.Button("Compare")
            with gr.Row():
                with gr.Column():
                    ia, ta, pa = gr.Image(label="Left"), gr.Markdown(), gr.Plot()
                with gr.Column():
                    ib, tb, pb = gr.Image(label="Right"), gr.Markdown(), gr.Plot()
            go.click(side, [a, b], [ia, ta, pa, ib, tb, pb])
            demo.load(side, [a, b], [ia, ta, pa, ib, tb, pb])
        with gr.Tab("Single variant"):
            v = gr.Dropdown(vs, value=vs[0], label="Reward variant")
            img, txt, plot = gr.Image(), gr.Markdown(), gr.Plot()
            v.change(single, v, [img, txt, plot])
        with gr.Tab("Results"):
            if (RESULTS / "ablation.md").exists():
                gr.Markdown((RESULTS / "ablation.md").read_text(encoding="utf-8"))
            for f in ("learning_curves.png", "tradeoff.png"):
                if (RESULTS / f).exists():
                    gr.Image(str(RESULTS / f), label=f)
    return demo


if __name__ == "__main__":
    build().launch()
