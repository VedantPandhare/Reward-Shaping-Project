# Lunar Lander Reward Shaping & Reward Hacking Lab

> **Audience:** a coding agent (and the human supervising it) that will build this project end to end.
> **Nature of project:** a portfolio / engineering project. Not a research paper. Rigor is used to avoid fooling ourselves, not to publish.
> **Cost:** free. Everything uses open-source tools and runs on a laptop CPU. Deployment uses a free tier.
> **Origin:** based on a project write-up by Hari (@harixganesh), extended with a multi-seed evaluation, ablations, a reward-hacking detector, a reusable library and a deployed demo.

---

## 0. TL;DR

Train a reinforcement-learning agent (PPO) to land a rocket in Gymnasium's `LunarLander-v3`. Then:

1. Write custom reward rules as a **wrapper** around the environment.
2. **Break it on purpose**: add a naive bonus that teaches the agent to hover forever instead of landing (**reward hacking**).
3. **Fix it** using improvement-based (potential-based) shaping plus a small per-frame time cost.
4. **Prove it** with multi-seed runs, an ablation table, and a numeric **hacking detector** (timeout rate, episode length).
5. **Package it**: a small reusable library, a README with results and GIFs, and a free deployed demo that shows the hacked and fixed agents side by side with a live reward breakdown.

The headline resume claim this project should support, with real numbers filled in from the experiments:

> "Diagnosed and fixed reward hacking in an RL agent: a naive shaping bonus made X% of episodes hover to timeout; an improvement-based reward cut that to Y% and restored a Z% landing rate across N seeds. Built a reusable detector and a deployed side-by-side demo."

---

## 1. Background: the concepts (read this first)

### 1.1 Reinforcement learning (RL)

An **agent** interacts with an **environment** in a loop:

1. The environment gives the agent an **observation** (what it can see).
2. The agent picks an **action**.
3. The environment moves forward one step and returns a **reward** (a number: good or bad) and the next observation.
4. Repeat until the episode ends.

The agent is never told what to do. It only gets reward numbers and must figure out which behavior earns the most total reward. It is like training a dog with treats, except treats are points. One full run from start to end is an **episode**; the sum of rewards in an episode is the **return** (the "score").

Key vocabulary:

| Term | Meaning |
|---|---|
| Environment | The game/simulator (here: Lunar Lander physics via Box2D) |
| Observation / state | The numbers the agent sees each step (8 numbers here) |
| Action | What the agent can do (4 discrete choices here) |
| Policy | The agent's decision rule: observation in, action out. Here it is a neural network |
| Reward | Number returned each step that says how good that step was |
| Episode | One attempt, from reset until it ends |
| Return | Total reward over an episode |
| Terminated | Episode ended naturally (landed, crashed, flew off-screen) |
| Truncated | Episode was cut off by a time limit (1,000 steps in Lunar Lander) |
| Discount factor (gamma) | How much future reward is valued versus immediate reward (close to 1 = patient) |

### 1.2 The Lunar Lander environment

- Gymnasium environment ID: **`LunarLander-v3`**. Older tutorials say `v2`; current Gymnasium only has `v3`.
- Runs at 50 frames per second; one agent decision per frame (one "step" = one frame).
- **Observation (8 numbers):**

| Index | Meaning | Example |
|---|---|---|
| 0 | x position relative to the pad (left/right) | 0.20 = a bit to the right |
| 1 | y position (height above the pad) | 0.90 = still high |
| 2 | x velocity (sideways speed) | -0.10 = drifting left |
| 3 | y velocity (up/down speed) | -0.50 = falling |
| 4 | angle (tilt, radians) | 0.05 = nearly upright |
| 5 | angular velocity (spin) | 0.00 = not spinning |
| 6 | left leg touching ground (0/1) | 0 |
| 7 | right leg touching ground (0/1) | 0 |

  The agent never sees the picture, only these numbers.

- **Actions (4, discrete):** 0 = do nothing, 1 = fire left side engine, 2 = fire main engine, 3 = fire right side engine.
- **Built-in reward (per frame):** the environment computes a "how good is this position" score:

  `position_score = -100*distance_to_pad - 100*speed - 100*|tilt| + 10*(left_leg_down) + 10*(right_leg_down)`

  The frame's reward is **the improvement** in this score since the previous frame, minus **0.3** if the main engine fired or **0.03** if a side engine fired (fuel cost). On the final frame the reward is replaced by **+100** for resting safely or **-100** for crashing.

  Worked example:

  | Frame | Distance | Speed | Tilt | Legs down | Position score |
  |---|---|---|---|---|---|
  | 1 | 0.50 | 0.40 | 0.10 | 0 | -50 - 40 - 10 = -100 |
  | 2 | 0.49 | 0.38 | 0.10 | 0 | -49 - 38 - 10 = -97 |

  Score improved by 3; main engine fired, so frame reward = 3 - 0.3 = **2.7**.

- **Solved** means an average return of **200 or more** over many episodes.
- Useful constructor options: `continuous=True` (continuous actions, needed for SAC/TD3), `gravity`, `enable_wind`, `wind_power`, `turbulence_power`. Verify exact argument names against the installed Gymnasium version.

### 1.3 Neural network policy

A neural network here is just a function: 8 numbers in, a preference score for each of the 4 actions out. In between are a few thousand adjustable numbers called **weights** ("dials"), random at the start (so the rocket flails). Stable-Baselines3's `MlpPolicy` gives a small standard network (two hidden layers of 64 units) that is fast on a CPU. You do not hand-build it.

### 1.4 PPO (Proximal Policy Optimization)

PPO is the training algorithm (OpenAI, 2017). The loop:

1. **Play**: run the current policy for a while and collect experience (observations, actions, rewards).
2. **Score**: estimate which actions turned out better than expected.
3. **Adjust**: nudge the weights toward the better actions.
4. Repeat.

PPO's one big idea is to **keep each update small** (it "clips" how far the new policy may move from the old one). That stops a single lucky stretch from wrecking what the agent already learned, which makes training stable and a good default choice. PPO is **on-policy**: it learns from data produced by the current policy.

### 1.5 Reward shaping

The reward function is the only thing that tells the agent what you want. **Reward shaping** means adding extra reward terms to guide the agent toward the behavior you want (for example, "reward being slow near the ground") or to speed up learning. Done well, shaping makes learning faster. Done badly, it teaches the agent something you did not intend (see reward hacking).

**Potential-based shaping** is the safe, principled form. Define a "potential" function `Phi(state)` that says how good a state is, and give extra reward equal to the **change** in potential:

`F(s, s') = gamma * Phi(s') - Phi(s)`

Ng, Harada & Russell (1999) showed this kind of shaping does not change which policy is optimal; it only helps the agent learn faster. The key property: the shaping terms **telescope** across an episode (they sum to roughly `Phi(end) - Phi(start)`), so there is nothing to farm by hanging around. Lunar Lander's built-in reward already has this "reward the improvement" structure.

### 1.6 Reward hacking (simple version)

> **The agent maximizes the reward you wrote, not the goal you had in mind.**

Analogy: you tell a cleaning robot "I'll pay you for every piece of trash you pick up." The robot learns to dump the bin and pick the trash up again, forever. It scores huge; the room is never clean. The robot is not being sneaky; your rule had a loophole.

In this project:

- **Goal in your head:** land softly on the pad.
- **Rule you write:** "+1 for every frame the rocket is upright and slow."
- **Loophole:** an attempt is cut off at 1,000 frames. A rocket that simply hovers can collect up to about 1,000 points from your rule and pay at most about 300 in fuel (0.3 x 1,000 if it fires the main engine every frame). **Landing ends the attempt**, which stops the payments. So hovering pays more than landing, and PPO learns exactly that.

Why it is hard to notice: **the score goes up while the behavior gets worse.** That is why we watch clips and log behavioral metrics, not just the score.

**The fix:** pay for *getting better* (change between frames, like the built-in reward) instead of for *sitting in a nice state*, and add a small fee every frame so waiting costs something. Hovering then earns nothing extra, and every idle frame costs a little, so finishing the landing becomes the best strategy.

---

## 2. Project goals and non-goals

### Goals
- Get a PPO agent landing reliably (average return >= 200 with the wrapper off).
- Build a custom reward wrapper with multiple reward variants.
- Demonstrate reward hacking, fix it, and quantify both with multi-seed evidence.
- Build a numeric hacking detector (not just eyeballing clips).
- Package the wrapper and detector as a reusable library.
- Ship a README with results and GIFs, plus a free deployed demo.

### Non-goals
- No novel algorithms, no paper-level claims, no hyperparameter search beyond a small sanity check.
- No GPU requirement.
- No paid services.

---

## 3. Tooling (all free / open source)

| Purpose | Tool |
|---|---|
| Language | Python 3.10+ |
| Environment | `gymnasium[box2d]` (Box2D physics; may need `swig` installed first) |
| RL algorithms | `stable-baselines3` (PPO main; DQN optional for algorithm swap; SAC optional on continuous version) |
| Deep learning backend | PyTorch (installed as a dependency of SB3; CPU build is fine) |
| Logging | TensorBoard (primary); Weights & Biases optional |
| Data / plots | `pandas`, `numpy`, `matplotlib` |
| Video / GIF | Gymnasium `RecordVideo` (needs `moviepy`) or `imageio` |
| Demo UI | `gradio` deployed on **Hugging Face Spaces** (free CPU tier); Streamlit Community Cloud is an alternative |
| Packaging | `pyproject.toml` (installable with `pip install -e .`) |
| Tests / lint | `pytest`, `ruff` |
| CI (optional) | GitHub Actions |
| Version control | Git + GitHub |

Free-tier limits change; check current terms for Hugging Face Spaces / Streamlit before relying on them.

**Version note:** Gymnasium 1.x and Stable-Baselines3 2.x must be compatible. Install the latest of both and pin the working versions into `requirements.txt` after the first successful run. Do not assume API details from memory; check the installed versions' docs when something differs.

---

## 4. Repository layout

```
lunar-reward-lab/
├── README.md                    # results table, GIFs, how to run, findings
├── pyproject.toml
├── requirements.txt             # pinned after first successful install
├── Makefile                     # make setup | smoke | train-all | eval | plots | demo
├── configs/
│   ├── base.yaml                # shared PPO hyperparameters, steps, eval settings
│   └── variants.yaml            # reward variants and their parameters
├── src/rewardlab/               # the reusable library
│   ├── __init__.py
│   ├── wrappers.py              # RewardShapingWrapper + variant reward functions
│   ├── detector.py              # HackingDetector (metrics + SB3 callback)
│   ├── envs.py                  # make_env(variant, seed, ...) helpers
│   ├── train.py                 # CLI: train one (variant, seed)
│   ├── evaluate.py              # CLI: evaluate a saved model with wrapper OFF
│   ├── record.py                # render GIF/MP4 of a model
│   └── analysis.py              # aggregate results, tables, plots
├── scripts/
│   ├── 00_install_check.py
│   ├── 01_random_agent.py       # watch it fail
│   ├── 02_train_baseline.py
│   └── run_all.py               # parallel sweep over variants x seeds
├── runs/                        # models, tensorboard logs (gitignored)
├── results/                     # CSVs, plots, GIFs committed to repo
├── app/
│   └── app.py                   # Gradio demo
└── tests/
    ├── test_wrappers.py
    └── test_detector.py
```

---

## 5. Implementation plan (phased, with acceptance criteria)

Priority tags: **P0** = must have, **P1** = should have (makes it impressive), **P2** = demo/polish, **P3** = stretch.

### Phase 1 (P0): Setup (under 1 hour)

Tasks:
1. Create a virtual environment (`python -m venv .venv`) with Python 3.10+.
2. Install `gymnasium[box2d]`, `stable-baselines3`, `tensorboard`, `moviepy`, `pandas`, `matplotlib`, `pytest`.
3. If Box2D install fails with an error mentioning **swig**, install swig first (`pip install swig`, or via the OS package manager), then reinstall the box2d extra.
4. Write `scripts/00_install_check.py` that imports all three libraries and creates `LunarLander-v3`, printing versions.

Acceptance: the script runs with no errors and prints versions.

### Phase 2 (P0): Watch it fail (20 minutes)

Tasks:
1. `scripts/01_random_agent.py`: create `LunarLander-v3` (with `render_mode="human"` locally, or `"rgb_array"` on a headless machine), loop with random actions using `reset()` and `step()`.
2. Each `step` returns `(observation, reward, terminated, truncated, info)`. Print the 8 observation numbers and match them to the table in section 1.2.
3. Report the average return of a random policy over 20 episodes (expect strongly negative, roughly -100 to -200).

Acceptance: random agent crashes repeatedly; observation numbers are printed and labeled.

### Phase 3 (P0): Train the baseline (about 20-30 minutes of wall-clock)

Tasks:
1. Train `PPO("MlpPolicy", env, ...)` on the built-in reward for ~1,000,000 timesteps.
2. Use vectorized environments (`make_vec_env` with `n_envs=16`) for speed. Suggested starting hyperparameters (these match the commonly published RL-Zoo settings for this task; treat them as a starting point and verify):

   | Parameter | Value |
   |---|---|
   | n_envs | 16 |
   | total_timesteps | 1,000,000 |
   | n_steps | 1024 |
   | batch_size | 64 |
   | n_epochs | 4 |
   | gamma | 0.999 |
   | gae_lambda | 0.98 |
   | ent_coef | 0.01 |
   | seed | set explicitly |

3. Enable TensorBoard logging. Watch **`ep_rew_mean`** (mean score of recent episodes). It should start well below zero, wobble, and climb.
4. Save the model (`model.save(...)`). Time the run and record it in the README.

Acceptance: average return over 10+ evaluation episodes is >= 200 (see Phase 4 for the proper evaluation protocol).

### Phase 4 (P0): Evaluate and record

**Evaluation protocol (applies to every variant, always):**
- Evaluate on the **raw environment with the reward wrapper OFF** so custom bonuses cannot inflate the score.
- Use at least **100 episodes** per model (minimum 10 for quick checks), `deterministic=True`, evaluation seeds different from training seeds.
- Record per-episode: return, episode length, whether it **terminated** or was **truncated** (timed out), whether it **landed** (final reward +100 / legs down and at rest), whether it **crashed** (final reward -100), and **fuel used** (count of main-engine frames x 0.3 plus side-engine frames x 0.03, or count actions directly).

Tasks:
1. `evaluate.py`: loads a model, runs the protocol, writes a CSV row per episode and returns a summary.
2. `record.py`: renders one or more episodes to GIF/MP4 (Gymnasium `RecordVideo` with `render_mode="rgb_array"` and `moviepy`, or `imageio`). Keep short GIFs (under about 10 seconds, small resolution) for the README.

Acceptance: baseline GIF showing a landing between the flags; evaluation CSV written.

### Phase 5 (P0): Custom reward wrapper

A **wrapper** is a thin layer around the environment that can change what comes out of it without touching the game code. Subclass `gymnasium.Wrapper` and override `step()`:

```python
class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, variant_cfg): ...
    def reset(self, **kw): ...   # reset any per-episode state (e.g. previous potential)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        extra, terms = self.variant_fn(obs, action, reward, terminated, truncated)
        info["reward_terms"] = {"env": reward, **terms}   # for breakdown + detector
        info["raw_reward"] = reward
        return obs, reward + extra, terminated, truncated, info
```

Requirements:
- Support variants selected by name from `configs/variants.yaml`.
- **Always** expose the unmodified environment reward as `info["raw_reward"]` and each custom term in `info["reward_terms"]` (this powers the detector and the live breakdown chart in the demo).
- Keep wrapper state (previous potential value) reset in `reset()`.
- Apply the wrapper to **training** environments only; evaluation uses the raw env.
- With vectorized envs, pass the wrapper via `make_vec_env(..., wrapper_class=...)` / `wrapper_kwargs`.

First variant to add: **time penalty** (a small negative reward every frame, the built-in has none). Retrain; it should still land.

Acceptance: unit tests show the wrapper changes reward as specified and does not change observations, termination, or truncation.

### Phase 6 (P0): Break it, then fix it

Implement these reward variants (all of them are *additions* on top of the built-in reward unless noted). Parameter values are starting points; tune lightly.

| ID | Name | Extra reward added to built-in | Expected behavior |
|---|---|---|---|
| V0 | `baseline` | none | Learns to land (>= 200) |
| V1 | `time_penalty` | `-0.05` per frame (try 0.01 to 0.1) | Still lands, maybe faster |
| V2 | `hover_hack` | `+1` per frame if `abs(angle) < 0.2` and `hypot(vx, vy) < 0.3` (obs[4], obs[2], obs[3]) | **Reward hacking**: hovers to the 1,000-frame timeout |
| V3 | `delta_fix` | Improvement-based: `c * (Phi_extra(s') - Phi_extra(s))` where `Phi_extra` = the same "upright and slow" quality as a smooth score (e.g. `-(abs(angle) + hypot(vx, vy))`), plus `-0.05` per frame | Lands; no hovering |
| V4 | `delta_no_time` | Same as V3 but without the time penalty | Ablation: does the time penalty matter? |
| V5 | `tilt_only_hack` | `+1` per frame for low tilt only (ignore position/speed) | Second hack: stays upright without landing, or lands away from the pad |
| V6 | `fuel_efficient` | Fuel cost multiplied (e.g. main engine `-0.6` instead of `-0.3` effective) | Trade-off: less fuel, possibly lower landing rate |
| V7 | `soft_landing` | Extra bonus/penalty on vertical speed and tilt at touchdown | Trade-off: gentler landings |

Notes:
- For V3, if using discount `gamma < 1`, use `F = gamma*Phi(s') - Phi(s)` for the policy-invariance guarantee. Set `Phi` of the terminal state to 0 (episodic convention). Keep scale `c` small enough that the shaping does not dominate the +/-100 terminal reward.
- The hover hack depends on magnitudes. Check that the per-frame bonus (+1) exceeds the maximum per-frame fuel cost (0.3) so hovering is net positive even with the main engine firing constantly. The hack may need a retrain or two (different seeds) to show; that is expected.
- **Watch the clip, not just the score.** In V2, `ep_rew_mean` can go up while the rocket hovers.

Acceptance: V2 reliably produces hovering (high timeout rate) in most seeds; V3 reliably lands (timeout rate near baseline). Both GIFs saved.

### Phase 7 (P1): Multi-seed runs

**Why seeds in a non-research project:** RL training is noisy. One run can succeed or fail by luck. Without repeats you cannot tell "my fix works" from "I got a lucky draw", and anyone rerunning your code may get a different result. Multiple seeds turn "it worked once for me" into "it worked 3 out of 3 times".

Tasks:
1. `scripts/run_all.py`: run every (variant, seed) pair, **3 seeds minimum (5 if time allows)**, in parallel processes (one per CPU core as available). Fix and record all seeds.
2. For each run save: model, TensorBoard logs, training-curve CSV.
3. Compute budget guide: ~15-25 min per 1M-step run on a laptop CPU. 8 variants x 3 seeds is roughly 24 runs; parallelism and 500k-step runs for non-critical variants (V5-V7) can reduce this. Baseline, V2 and V3 should use the full 1M steps.
4. Plot **mean +/- std** of the learning curve across seeds, one plot per variant and one overlay of V0/V2/V3.

Acceptance: `results/learning_curves.png` exists with shaded std bands; every run is reproducible from the seed in its config.

### Phase 8 (P1): Hacking detector

A numeric detector so hacking is flagged by metrics rather than eyeballing. Implement `HackingDetector` as both a standalone analyzer (on evaluation logs) and an SB3 `BaseCallback` (during training).

Metrics (computed per rolling window of episodes):
- **Timeout rate:** fraction of episodes with `truncated=True` (hit the 1,000-frame limit). Hovering shows up as a spike.
- **Mean episode length:** hovering pushes it toward 1,000.
- **Custom-term share:** `sum(custom reward terms) / sum(|total reward|)`. If your own bonus dominates the return, that is a red flag.
- **Return-vs-landing divergence:** training return rising while evaluation landing rate (wrapper off) falls or stays flat.

Behavior:
- Log all metrics to TensorBoard under a `hack/` prefix.
- Emit a warning (and a flag in the results CSV) when thresholds are crossed (defaults: timeout rate > 30% and custom-term share > 50%; make configurable).
- Unit test with synthetic episodes (all-timeouts should trigger; normal landings should not).

Acceptance: detector flags V2 and V5 runs and does not flag V0 and V3 runs on the saved results.

### Phase 9 (P1): Ablation table and results

Produce `results/ablation.csv` and a Markdown table for the README. For each variant, report mean +/- std across seeds of the following (all from raw-env evaluation, 100 episodes per seed):

| Metric | Meaning |
|---|---|
| Mean return | Average raw score |
| Landing rate | Fraction of episodes that land safely |
| Crash rate | Fraction that crash |
| Timeout rate | Fraction that hit 1,000 frames (hacking signal) |
| Mean episode length | Frames per episode |
| Mean fuel used | Fuel cost per episode |
| Steps to reach 200 | Sample efficiency (if reached) |
| Hack flag | From the detector |

Also produce:
- A **trade-off scatter** (fuel used vs. landing rate) across V0, V1, V6, V7.
- A short written **findings** section: what the fix does, which part (delta structure vs. time penalty) matters (V3 vs V4), whether the failure mode generalizes (V5), and what surprised you.

### Phase 10 (P1): Algorithm swap (optional but strong)

Run the best and worst rewards (V3 and V2) under a second algorithm, e.g. **DQN** (discrete) or **SAC** (needs `continuous=True` env). 3 seeds each. Question to answer: does the hack appear under both algorithms (a reward problem) or only under PPO (an algorithm problem)? Add to the results table.

### Phase 11 (P2): Packaging as a library

Turn `src/rewardlab` into a pip-installable package (`pip install -e .`):
- `RewardShapingWrapper` that works on any Gymnasium environment given a reward function.
- `HackingDetector` (standalone + SB3 callback) with configurable thresholds.
- A short usage example in the README (5-10 lines) showing how to wrap any env and attach the callback.
- Tests passing in CI (GitHub Actions optional).

### Phase 12 (P2): Deployed demo (Gradio on Hugging Face Spaces)

Features, in priority order:
1. **Variant picker:** choose a reward variant (baseline / hover hack / fixed) and watch that trained agent fly, with its stats (return, landed or not, frames, fuel).
2. **Side-by-side:** hacked agent and fixed agent shown at the same time (two videos or one composite). This is the main "five-second understanding" feature.
3. **Live reward breakdown:** a stacked chart of per-frame reward by term (distance, speed, tilt, legs, fuel, your bonus) next to the video, from the `info["reward_terms"]` data. It should visibly show the bonus term dominating during hovering.
4. **Results tab:** the ablation table and learning curves.

Deployment guidance:
- Commit small trained models (the policy is tiny) and **pre-rendered GIFs/MP4s** to the Space repo. Pre-rendering is the most reliable on a free CPU tier; live inference is optional because the model is small, but Box2D rendering on a headless server needs `rgb_array` mode.
- Keep the Space's `requirements.txt` minimal and pinned.
- Verify the Space builds and loads from a clean checkout.

### Phase 13 (P3): Stretch ideas

- **Play against the AI:** let a person fly the lander and compare to the agent. Locally this is easy with Gymnasium's play utility (keyboard). On the web, keyboard control inside Gradio is awkward; use button-driven stepping, or keep it local-only and say so.
- **Robustness test:** evaluate each variant's policy with wind and turbulence enabled (`enable_wind`) without retraining; report which rewards generalize.
- **Weights & Biases** dashboard with public report.
- **Blog post:** a short write-up with the before/after GIFs, numbers and what surprised you.

---

## 6. Reproducibility requirements

- Every run is defined by `(variant, seed, config)`; seeds set for PPO (`seed=`), the environment (`reset(seed=...)`), NumPy and PyTorch.
- Training seeds and evaluation seeds must differ.
- `make smoke` runs a tiny 20k-step training for every variant to confirm the pipeline works end to end in a few minutes.
- `make train-all`, `make eval`, `make plots`, `make demo` each work from a clean clone after `make setup`.
- Pin dependency versions once a working set is found.

---

## 7. Success criteria (definition of done)

- [ ] Baseline (V0) averages >= 200 return, wrapper off, over >= 100 episodes, for most seeds.
- [ ] Hover hack (V2) demonstrably causes hovering: timeout rate and episode length are clearly higher than baseline, shown across seeds, with a GIF.
- [ ] Fix (V3) restores landing: landing rate close to baseline, timeout rate near baseline, with a GIF.
- [ ] Ablation table covers all variants with mean +/- std across seeds.
- [ ] Hacking detector flags the hacked variants and not the healthy ones.
- [ ] README has: overview, results table, learning-curve plots, GIFs, findings, how to reproduce.
- [ ] Library installs with `pip install -e .` and tests pass.
- [ ] Demo is deployed on a free tier and loads from the README link.
- [ ] All results come from raw-env evaluation, not training reward.

---

## 8. Common pitfalls

- **Box2D install fails mentioning "swig":** install swig first, then reinstall the box2d extra.
- **`LunarLander-v2` not found:** use `LunarLander-v3`.
- **Trusting a single run:** always repeat across seeds before calling a reward good or bad.
- **Evaluating with the wrapper on:** inflates scores with your own bonus. Always evaluate raw.
- **Judging only by score:** hacked agents can have a rising score. Check timeout rate, episode length and clips.
- **Hack does not appear:** try another seed, make sure the bonus exceeds max fuel cost, or raise the bonus magnitude. The doc notes it may take a retrain or two.
- **Fix overshoots:** time penalty too large can make the agent rush and crash; sweep 0.01 to 0.1.
- **Headless machine rendering errors:** use `render_mode="rgb_array"` and record video instead of opening a window.
- **Vectorized env wrappers:** wrapper state must be per-environment and reset on `reset()`.

---

## 9. Deliverables checklist for the agent

1. Working repo matching section 4, with `README.md`.
2. Trained models and evaluation CSVs for all variants and seeds (models may be Git LFS or released separately if large; the policies are small).
3. `results/` containing: learning curves, ablation table, trade-off plot, GIFs for baseline, hover hack and fix.
4. Installable `rewardlab` package with tests.
5. Deployed Gradio demo (link in README).
6. A short findings write-up in the README.

---

## 10. Resume framing (lead with product impact)

Suggested bullets, to be filled with real numbers:

- "Built and deployed a reward-debugging lab for RL: showed a naive shaping bonus made X% of episodes hover to timeout, and an improvement-based reward cut that to Y% while restoring a Z% landing rate across N seeds."
- "Shipped a reusable reward-hacking detector (timeout rate, episode length, bonus share) as a pip-installable library and a free live demo comparing hacked vs. fixed agents side by side."
- Technical detail (PPO, Gymnasium, Stable-Baselines3, potential-based shaping, multi-seed ablations) follows after the impact.

---

## 11. References (start here, in this order)

1. Stable-Baselines3: Install: https://stable-baselines3.readthedocs.io/en/master/guide/install.html
2. Gymnasium: Basic Usage: https://gymnasium.farama.org/introduction/basic_usage/
3. Gymnasium: Lunar Lander: https://gymnasium.farama.org/environments/box2d/lunar_lander/
4. Hugging Face Deep RL Course, Unit 1 (trains PPO on this game; it says v2, use v3): https://huggingface.co/learn/deep-rl-course/unit1/introduction
5. Stable-Baselines3: Examples (training, saving, loading, GIFs): https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
6. Gymnasium: Implementing Custom Wrappers: https://gymnasium.farama.org/tutorials/gymnasium_basics/implementing_custom_wrappers/
7. Ng, Harada, Russell (1999), "Policy invariance under reward transformations: Theory and application to reward shaping" (the theory behind potential-based shaping).
8. Schulman et al. (2017), "Proximal Policy Optimization Algorithms" (PPO).

---

## 12. Glossary

- **Action:** a choice the agent makes each frame (4 in Lunar Lander).
- **Callback (SB3):** code hooked into training to log or react to events.
- **Episode:** one attempt from reset to end.
- **Gymnasium:** open-source library of RL environments (successor to OpenAI Gym).
- **On-policy:** learns from data generated by its current policy (PPO).
- **Potential-based shaping:** shaping reward equal to the change in a state-quality function; does not change the optimal policy.
- **PPO:** Proximal Policy Optimization; stable policy-gradient training with small clipped updates.
- **Reward hacking:** the agent exploits a loophole in the reward and scores high without achieving the intended goal.
- **Reward shaping:** adding extra reward terms to guide or speed up learning.
- **Seed:** the starting value for random number generators; fixing it makes a run repeatable.
- **Truncated:** episode ended by a time limit rather than a natural ending.
- **Wrapper:** a layer around an environment that modifies its inputs or outputs without changing its code.
