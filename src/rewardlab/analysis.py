"""Aggregate runs into the ablation table, learning curves and trade-off plot.

    python -m rewardlab.analysis            # reads runs/, writes results/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from rewardlab.envs import load_variants  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
METRICS = ["mean_return", "landing_rate", "crash_rate", "timeout_rate", "mean_length", "mean_fuel",
           "custom_share", "steps_to_200"]


def steps_to_200(episodes: pd.DataFrame, window: int = 100, target: float = 200.0):
    """First training timestep where the rolling mean raw return (stochastic policy) >= target."""
    if len(episodes) < window:
        return np.nan
    roll = episodes["raw_return"].rolling(window).mean()
    hit = roll[roll >= target]
    return float(episodes.loc[hit.index[0], "timesteps"]) if len(hit) else np.nan


def collect(algo_suffix: str = "") -> pd.DataFrame:
    rows = []
    for vdir in sorted(RUNS.glob("*")):
        if vdir.name.startswith("_"):
            continue
        name = vdir.name
        if (algo_suffix and not name.endswith(algo_suffix)) or (not algo_suffix and "_dqn" in name):
            continue
        variant = name[: -len(algo_suffix)] if algo_suffix else name
        for sdir in sorted(vdir.glob("seed*")):
            f = sdir / "eval_summary.json"
            if not f.exists():
                continue
            row = {"variant": variant, "seed": int(sdir.name[4:]), **json.loads(f.read_text())}
            ep = sdir / "episodes.csv"
            row["steps_to_200"] = steps_to_200(pd.read_csv(ep)) if ep.exists() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def ablation(df: pd.DataFrame) -> pd.DataFrame:
    order = [v for v in load_variants() if v in set(df["variant"])]
    out = []
    for v in order:
        d = df[df["variant"] == v]
        row = {"variant": v, "n_seeds": len(d)}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_std"] = d[m].mean(), d[m].std(ddof=0)
        row["hack_flag"] = f"{int(d['hack_flag'].sum())}/{len(d)}"
        row["reached_200"] = f"{int(d['steps_to_200'].notna().sum())}/{len(d)}"
        out.append(row)
    return pd.DataFrame(out)


def to_markdown(ab: pd.DataFrame) -> str:
    def pm(r, m, fmt):
        return f"{r[m + '_mean']:{fmt}} ± {r[m + '_std']:{fmt}}"

    lines = ["| Variant | Seeds | Return | Landing | Crash | Timeout | Ep. length | Fuel | "
             "Steps to 200 | Hack flag |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in ab.iterrows():
        s200 = (f"{r['steps_to_200_mean'] / 1e3:.0f}k ± {r['steps_to_200_std'] / 1e3:.0f}k "
                f"({r['reached_200']})" if r["reached_200"][0] != "0" else f"never ({r['reached_200']})")
        lines.append(f"| {r['variant']} | {r['n_seeds']} | {pm(r, 'mean_return', '.0f')} | "
                     f"{pm(r, 'landing_rate', '.2f')} | {pm(r, 'crash_rate', '.2f')} | "
                     f"{pm(r, 'timeout_rate', '.2f')} | {pm(r, 'mean_length', '.0f')} | "
                     f"{pm(r, 'mean_fuel', '.0f')} | {s200} | {r['hack_flag']} |")
    return "\n".join(lines)


def load_curves(variant: str, key: str = "raw_return", n_points: int = 200, window: int = 100):
    """Per-seed smoothed curve interpolated onto a common timestep grid."""
    curves, grid_max = [], 0
    for sdir in sorted((RUNS / variant).glob("seed*")):
        f = sdir / "episodes.csv"
        if not f.exists():
            continue
        ep = pd.read_csv(f)
        ep["smooth"] = ep[key].rolling(window, min_periods=10).mean()
        curves.append(ep.dropna(subset=["smooth"]))
        grid_max = max(grid_max, curves[-1]["timesteps"].max())
    if not curves:
        return None, None
    grid = np.linspace(0, min(c["timesteps"].max() for c in curves), n_points)
    ys = np.stack([np.interp(grid, c["timesteps"], c["smooth"]) for c in curves])
    return grid, ys


def plot_curves(variants: list[str], path: Path, key: str = "raw_return", title: str = "",
                ylabel: str = "raw return (rolling mean of 100 episodes)"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in variants:
        grid, ys = load_curves(v, key)
        if grid is None:
            continue
        m, s = ys.mean(0), ys.std(0)
        ax.plot(grid, m, label=f"{v} (n={len(ys)})")
        ax.fill_between(grid, m - s, m + s, alpha=0.2)
    ax.axhline(200, color="gray", ls="--", lw=0.8)
    ax.set(xlabel="training timesteps", ylabel=ylabel, title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_tradeoff(df: pd.DataFrame, path: Path, variants=("baseline", "time_penalty",
                                                           "fuel_efficient", "soft_landing")):
    fig, ax = plt.subplots(figsize=(6, 5))
    for v in variants:
        d = df[df["variant"] == v]
        if d.empty:
            continue
        ax.scatter(d["mean_fuel"], d["landing_rate"], alpha=0.5)
        ax.errorbar(d["mean_fuel"].mean(), d["landing_rate"].mean(), xerr=d["mean_fuel"].std(ddof=0),
                    yerr=d["landing_rate"].std(ddof=0), fmt="o", capsize=3, label=v)
    ax.set(xlabel="mean fuel used per episode", ylabel="landing rate (raw-env eval)",
           title="Fuel vs landing trade-off (dots = seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    df = collect()
    if df.empty:
        print("no evaluated runs found")
        return
    df.to_csv(RESULTS / "per_seed.csv", index=False)
    ab = ablation(df)
    ab.to_csv(RESULTS / "ablation.csv", index=False)
    (RESULTS / "ablation.md").write_text(to_markdown(ab) + "\n")
    print(to_markdown(ab))
    present = set(df["variant"])
    for v in present:
        plot_curves([v], RESULTS / "curves" / f"{v}.png", title=v)
    plot_curves([v for v in ("baseline", "hover_hack", "delta_fix") if v in present],
                RESULTS / "learning_curves.png", title="Raw return: baseline vs hack vs fix")
    plot_curves([v for v in ("baseline", "hover_hack", "delta_fix") if v in present],
                RESULTS / "learning_curves_shaped.png", key="shaped_return",
                title="What the agent optimises (shaped return)", ylabel="shaped return")
    plot_tradeoff(df, RESULTS / "tradeoff.png")


if __name__ == "__main__":
    main()
