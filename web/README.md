---
title: Lunar Lander Reward-Hacking Lab
emoji: 🚀
colorFrom: red
colorTo: indigo
sdk: static
pinned: false
short_description: Play against PPO agents trained on hacked vs fixed rewards
---

# Lunar Lander Reward-Hacking Lab (3D demo)

Static site: Three.js rendering, a planck.js (Box2D) port of Gymnasium's `LunarLander-v3`, and the real
PPO policies exported as JSON. Fly the lander yourself against an agent, or watch the **hover-hack**
agent (trained with a naive per-frame bonus) next to the **delta-fix** agent on the same terrain.

Run locally: `python -m http.server -d web 8000` then open http://localhost:8000.
Source and training code: the `lunar-reward-lab` repository.
