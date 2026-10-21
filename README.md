# Lunar Lander Reward Shaping & Reward Hacking Lab

A naive per-frame shaping bonus taught a PPO agent to **hover until timeout instead of landing**
(100% of evaluation episodes timed out, 0% landed). An improvement-based (potential) reward fixed it:
**99% landing, 0% timeouts**, across 3 seeds. A numeric detector flags the hacked runs and none of the
healthy ones, and the whole thing ships as a small library plus a side-by-side demo.

| Hover hack (naive bonus) | Fix (improvement-based) | Baseline |
|---|---|---|
| ![hack](results/gifs/hover_hack.gif) | ![fix](results/gifs/delta_fix.gif) | ![base](results/gifs/baseline.gif) |

![reward breakdown](results/breakdown_hover_hack.png)
*Per-frame reward of the hacked agent: the bonus term (purple) dwarfs everything the environment pays.*

## Results

PPO, `LunarLander-v3`, 3 training seeds (0-2), 1M steps (500k for the last three variants).
**All numbers are from the raw environment reward**, 100 deterministic episodes per seed, evaluation
seeds (10000+) disjoint from training seeds. Mean ± std across seeds.

| Variant | Seeds | Return | Landing | Crash | Timeout | Ep. length | Fuel | Steps to 150 (train) | Hack flag |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 3 | 256 ± 3 | 0.98 ± 0.02 | 0.00 ± 0.00 | 0.02 ± 0.02 | 365 ± 17 | 53 ± 3 | 965k ± 32k (3/3) | 0/3 |
| time_penalty | 3 | 265 ± 1 | 1.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 305 ± 7 | 47 ± 1 | 744k ± 75k (3/3) | 0/3 |
| **hover_hack** | 3 | -78 ± 22 | 0.00 ± 0.00 | 0.00 ± 0.00 | **1.00 ± 0.00** | 1000 ± 0 | 184 ± 0 | never (0/3) | **3/3** |
| **delta_fix** | 3 | 258 ± 7 | 0.99 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 347 ± 33 | 51 ± 6 | 868k ± 68k (3/3) | 0/3 |
| delta_no_time | 3 | 266 ± 3 | 0.99 ± 0.01 | 0.01 ± 0.01 | 0.00 ± 0.00 | 279 ± 14 | 41 ± 3 | 649k ± 40k (3/3) | 0/3 |
| tilt_only_hack | 3 | -64 ± 23 | 0.00 ± 0.00 | 0.00 ± 0.00 | 1.00 ± 0.00 | 997 ± 4 | 184 ± 1 | never (0/3) | 3/3 |
| fuel_efficient* | 3 | 179 ± 38 | 0.77 ± 0.06 | 0.14 ± 0.02 | 0.09 ± 0.04 | 427 ± 132 | 61 ± 27 | never (0/3) | 0/3 |
| soft_landing* | 3 | 167 ± 21 | 0.79 ± 0.13 | 0.08 ± 0.04 | 0.13 ± 0.10 | 621 ± 78 | 97 ± 22 | never (0/3) | 0/3 |

\* trained for 500k steps only, so these are not directly comparable with the 1M-step rows.
"Steps to 150" is the first training step where the rolling-100 *training* return (stochastic policy)
reaches 150; training returns sit below deterministic evaluation, so 200 is rarely reached there.
Full tables: [`results/ablation.csv`](results/ablation.csv), [`results/per_seed.csv`](results/per_seed.csv).

![learning curves](results/learning_curves.png)
![tradeoff](results/tradeoff.png)

## Findings

- **The hack is reliable.** `hover_hack` (+1 per frame when upright and slow) hovered to the 1000-frame
  timeout in 3/3 seeds. Its deterministic policy fires the engine for ~184 fuel and collects the bonus
  rather than landing. Note the training curve: raw return still climbs to ~90 because the
  *stochastic* policy sometimes lands by accident; the deterministic policy is what hovers.
  Score alone would not have shown the failure; timeout rate and clips do.
- **The fix works.** `delta_fix` (scale × change in an upright-and-slow potential, plus a 0.05 time
  penalty) matches the baseline: 99% landing, 0% timeouts, similar fuel. The shaping telescopes, so
  hovering earns nothing.
- **Delta structure matters; the time penalty did not (here).** `delta_no_time` landed as reliably as
  `delta_fix` (and was a bit faster/leaner), so the improvement-based form is what removes the exploit.
  The time penalty is not needed for this environment at this scale. `time_penalty` alone also helped
  slightly (shorter, cheaper landings).
- **The failure generalises.** `tilt_only_hack` (bonus for low tilt only) failed the same way: 100%
  timeouts. Hovering upright is the cheapest way to farm any per-frame state bonus.
- **Detector.** Flags (timeout rate > 30% *and* bonus share of return > 50%) fired on all 3 seeds of both
  hacks, and on none of the 18 other runs. Requiring both conditions matters: a baseline that merely
  hovers early in training has timeouts but no bonus share. During training the callback raised the
  warning around 0.5-0.65M steps for hover_hack.
- **Trade-offs.** Doubling the fuel cost (`fuel_efficient`) and a touchdown penalty (`soft_landing`)
  both lowered landing rate at 500k steps. Landing is also frequently *off the pad*: pad-landing rate was
  0.45 and 0.66 respectively, vs 0.98 for baseline (`results/per_seed.csv`, `on_pad_rate`). "Landed" means
  the environment's +100 (lander at rest), not necessarily on the pad.
- **Caveat.** 3 seeds per variant; stds are small but this is not a statistical study.

## Algorithm swap: DQN

Same rewards, DQN instead of PPO (500k steps, 3 seeds, one env, hyper-parameters from the common
RL-Zoo setting, not tuned). Raw-env evaluation, 100 episodes per seed.

| Variant | Return | Landing | Crash | Timeout | Ep. length | Hack flag |
|---|---|---|---|---|---|---|
| hover_hack (DQN) | -231 ± 375 | 0.27 ± 0.28 | 0.50 ± 0.35 | 0.23 ± 0.28 | 413 ± 251 | 1/3 |
| delta_fix (DQN) | 166 ± 29 | 0.75 ± 0.10 | 0.21 ± 0.09 | 0.04 ± 0.04 | 369 ± 177 | 0/3 |

Per seed, `hover_hack`: seed 0 hovers (63% timeouts, bonus share 0.50, flagged), seed 1 diverged
into constant crashes (return -750), seed 2 mostly lands (66%, timeout 6%). **Reading:** the hack is
not PPO-only (one DQN seed shows the same hovering and the detector flags it), but it is far less
consistent than under PPO, and DQN is a weaker, less stable learner here overall (even `delta_fix` lands
only 75%). With an untuned 500k-step DQN and 3 seeds this is suggestive, not conclusive. Files:
[`results/ablation_dqn.md`](results/ablation_dqn.md).

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -e ".[dev,demo]"          # needs swig if Box2D fails to build
python scripts/00_install_check.py
python scripts/01_random_agent.py     # random policy: ~ -210 return
python scripts/run_all.py             # all variants x seeds (about 20-40 min with 16 cores)
python scripts/run_all.py --algo dqn --variants hover_hack delta_fix --steps 500000   # ~1.5 h
python -m rewardlab.analysis          # results/ tables + plots
python scripts/make_gifs.py           # GIFs + breakdown JSON
python app/app.py                     # demo
pytest -q
```
One run: `python -m rewardlab.train --variant hover_hack --seed 0`. TensorBoard: `tensorboard --logdir runs`
(detector metrics under `hack/`). Pinned versions are in `requirements.txt`. `make smoke` runs 20k-step
runs for every variant. Baseline 1M-step training took ~19.5 min per run single-threaded (9 in parallel).

## Library

```python
import gymnasium as gym
from stable_baselines3 import PPO
from rewardlab import RewardShapingWrapper, HackingCallback, make_reward_fn

env = RewardShapingWrapper(gym.make("LunarLander-v3"), make_reward_fn("hover_hack"))
model = PPO("MlpPolicy", env)
model.learn(100_000, callback=HackingCallback(episodes_path="episodes.csv"))
```
Any callable `reward_fn(obs, action, reward, terminated, truncated) -> (extra, {term: value})` works.
The wrapper never changes observations or termination, always exposes `info["raw_reward"]` and
`info["reward_terms"]`. `HackingDetector.analyze(df)` runs the same metrics on an evaluation log.

## Layout

`src/rewardlab/` library · `configs/` hyper-parameters and variants · `scripts/` runnable steps ·
`app/` Gradio demo (pre-rendered GIFs + breakdown JSON in `app/assets/`) · `results/` committed outputs ·
`tests/` wrapper and detector tests. Spec: [`lunar_lander_reward_shaping_project.md`](lunar_lander_reward_shaping_project.md).

## Demo deployment (Hugging Face Spaces)

`app/` is self-contained: copy `app/app.py` and `app/assets/` to a Gradio Space with a minimal
`requirements.txt` (`gradio`, `matplotlib`, `numpy`). **Not deployed yet**, and no live link exists.
