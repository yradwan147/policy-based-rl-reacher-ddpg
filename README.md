# Policy-Based RL — Reacher (DDPG)

Final project for Udacity's *Deep Reinforcement Learning Nanodegree*
(nd893) — Continuous Control. A from-scratch DDPG agent that solves
Unity ML-Agents' Reacher environment (20-arm version): each arm
is rewarded +0.1 for every timestep its hand is inside the moving
target sphere; the environment is solved when the average score
across all 20 arms exceeds +30 over 100 consecutive episodes.

## Files

```
model.py     Actor + Critic networks (both with BatchNorm on the first hidden layer)
agent.py     DDPGAgent + ReplayBuffer + OUNoise
train.py     CLI training loop
```

## Architecture

* **Actor**: 33 → 256 → BN → 128 → 4, `tanh` output → bounded action.
* **Critic**: state goes 33 → 256 → BN, then `cat([h, action])` →
  128 → 1.
* **Soft target updates** (τ = 1e-3) after every learning step.
* **Multi-arm experience** — every step adds 20 transitions to the
  shared replay buffer; we run 10 minibatch updates every 20
  environment steps (the rate that keeps Reacher stable).
* **OU noise** with shape `(num_agents, action_size)` so every arm
  gets its own correlated exploration trajectory.

## Hyper-parameters

| Param | Value |
|---|---|
| Buffer | 1 000 000 |
| Batch | 128 |
| Gamma | 0.99 |
| Tau | 1e-3 |
| Actor / Critic LR | 1e-4 / 1e-3 |
| Update every / num updates | 20 / 10 |
| Critic gradient clip | 1.0 |

## Running

```bash
# 1. Download the Reacher (multi-agent) Unity env per the rubric.
pip install torch numpy unityagents
python3 train.py --env path/to/Reacher_Linux/Reacher.x86_64 --episodes 300
```

Solved networks are saved as `reacher_actor.pth` and `reacher_critic.pth`.

## Standing-out work

* The 20-arm replay buffer is fed in parallel — 20× the samples per
  environment step compared to single-agent DDPG.
* Critic gradient clipping is on by default; without it the loss
  explodes on the second rollout when the buffer is still half-full.
* Update frequency tuning (every 20 steps, 10 minibatch updates) is
  the sweet spot from the project readme — exposed as `update_every`
  / `num_updates` constructor args for quick ablation.

## License

Educational submission for Udacity nd893. Unity Reacher env © Unity / Udacity.
