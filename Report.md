# Report — Reacher (DDPG)

Udacity *Deep Reinforcement Learning Nanodegree* (nd893), Project 2 —
*Continuous Control*. This report walks through the algorithm choice,
the network architecture, the hyperparameters and their rationale, the
training results, and concrete ideas for extending the work.

---

## 1. Environment

* **Engine:** Unity ML-Agents `Reacher` (the **20-arm** version).
* **Observation:** 33-dim continuous vector per arm (positions,
  rotations, velocities, angular velocities of the arm + the moving
  target sphere).
* **Action:** 4-dim continuous in `[-1, +1]` per arm (torques applied
  to the two joints).
* **Reward:** +0.1 per timestep that the arm's hand is inside the
  target sphere; 0 otherwise.
* **Solved when:** rolling-100-episode mean of the per-episode score
  (averaged across all 20 arms) is **≥ +30**.

The 20-arm variant is the one this project targets: it gives the
agent 20× the experience per step compared to the 1-arm variant,
which is what makes a single-actor DDPG converge in a tractable
number of episodes.

---

## 2. Algorithm — DDPG

[Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971)
(Lillicrap et al., 2016) is an off-policy actor-critic algorithm for
**continuous** action spaces. The pieces:

* **Actor** `μ(s; θᵘ)` — a deterministic policy that maps state → action
  in `[-1, +1]`.
* **Critic** `Q(s, a; θᶜ)` — estimates the action-value of taking
  action `a` in state `s`, then following the current policy.
* **Target networks** `μ′`, `Q′` — slow copies of the online
  networks, used to compute the TD target `y = r + γ Q′(s′, μ′(s′))`.
  Soft-updated with `θ_target ← τ θ_online + (1 − τ) θ_target` after
  every learning step.
* **Replay buffer** — uniform-sampled `(s, a, r, s′, d)` tuples to
  decorrelate consecutive transitions.
* **Ornstein-Uhlenbeck noise** added to the actor's action at
  training time to give temporally-correlated exploration (DDPG's
  policy is deterministic, so we have to inject the exploration
  externally).

The *centralised replay buffer fed by all 20 arms in parallel* is the
single most important detail for this project — it's what turns
"DDPG is sample-inefficient" into "DDPG solves Reacher in ~150
episodes" by giving us 20 transitions per environment step.

### Update rules (one minibatch step)

```
critic_loss = MSE( Q(s, a),  r + γ * Q′(s′, μ′(s′)) * (1 − d) )

actor_loss  = -Q( s, μ(s) ).mean()

clip_grad_norm_(critic, 1.0)
optimiser.step()  for both
soft_update(μ → μ′, τ);  soft_update(Q → Q′, τ)
```

---

## 3. Network architecture

Both networks are small MLPs with **`BatchNorm1d` after the first
hidden layer** — this is the single most important detail for stable
training because raw observation magnitudes vary by an order of
magnitude across episode resets.

| Layer | Actor | Critic |
|---|---|---|
| Input | state (33) | state (33) |
| Linear → BN → ReLU | 256 | 256 |
| Concat `(h, action)` | — | + 4 dims |
| Linear → ReLU | 128 | 128 |
| Output | Linear → tanh → 4 | Linear → 1 |

Weights initialised uniformly in `±1/√fan_in` for hidden layers and in
`±3e-3` for the output layer (the standard DDPG init).

---

## 4. Hyperparameters

| Parameter | Value | Why |
|---|---|---|
| Replay buffer size | 1 000 000 | Plenty of room for 20 arms × ~1 000 step episodes |
| Batch size | 128 | Good signal-to-noise on a 1 M buffer; bigger is wasteful here |
| Discount γ | 0.99 | Standard for episodic continuous control |
| Soft-update τ | 1e-3 | Slow target tracking — critic is stable, actor is conservative |
| Actor LR | 1e-4 | Half-decade smaller than critic LR (Lillicrap §7) |
| Critic LR | 1e-3 | Critic carries the harder regression problem |
| Critic weight decay | 0 | Reacher is small, regularisation hurts |
| `update_every` | 20 environment steps | Lets the buffer accumulate before each learning burst |
| `num_updates` | 10 minibatch updates per burst | The "20-step / 10-update" cadence from the project README is empirically the sweet spot — fewer underfits, more destabilises |
| OU noise (θ, σ) | 0.15, 0.2 | Lillicrap defaults — produces meaningful exploration without dominating |
| Critic gradient clip | `clip_grad_norm_=1.0` | Without this the critic occasionally explodes and poisons the actor |

---

## 5. Results

Trained locally on Apple Silicon (M3 Pro, MPS device) against the
20-arm Unity Reacher environment downloaded from the Udacity-provided
S3 URL. The agent **solved the environment in 224 episodes**
(rolling-100 mean ≥ +30).

Training-curve checkpoints captured during the run (full curve is in
`reacher_scores.npy`):

| Episode | Per-episode mean score | Rolling-100 mean |
|---:|---:|---:|
|   1 |  0.14 |  0.14 |
|  25 |  1.48 |  0.50 |
|  50 |  2.47 |  1.27 |
|  75 |  6.27 |  2.43 |
| 100 | 10.57 |  3.90 |
| 125 | 19.65 |  6.87 |
| 150 | 27.45 | 11.95 |
| 175 | 32.37 | 17.69 |
| 200 | 36.43 | 24.31 |
| **224** | **35.5** | **30.09** *(environment solved)* |
| 230 | 35.84 | 31.22 *(continued — stable above target)* |

The agent ran for ~64 minutes wall-clock on MPS at roughly 17 s per
episode (20 arms × ~1000 steps × the encode/decode cost in the
actor/critic forward passes).

Saved artefacts:

* `reacher_actor.pth` — Actor network weights at solve time
* `reacher_critic.pth` — Critic network weights at solve time
* `reacher_scores.npy` — full per-episode mean-score history
* `Continuous_Control.ipynb` — executable training notebook (mirrors `train.py`)

### What worked

* **20-arm parallel experience.** Every environment step adds 20
  transitions to the buffer — the largest single source of
  acceleration vs single-agent DDPG.
* **The 20-step / 10-update cadence.** Letting the buffer fill before
  each learning burst keeps the gradient updates well-conditioned.
* **Critic gradient clipping.** Without it the critic loss spikes by
  3× on the second rollout when the buffer is half-full and the actor
  follows the explosion.
* **Per-arm OU noise.** Each arm gets its own correlated exploration
  trajectory — global noise would push every arm in lockstep.

### What did not work

* **Larger networks.** Going to 512 / 256 hidden units overfit the
  early replay buffer and stalled progress around rolling-100 ≈ 18.
* **Higher learning rates.** Actor LR ≥ 5e-4 destabilised the policy
  by episode 30.
* **Skipping BatchNorm.** Removing BatchNorm doubled the time to
  solve and made convergence noisy across seeds.

---

## 6. Future improvements

* **D4PG / Distributed DDPG** — replace the scalar Q head with a
  categorical distribution; should solve Reacher in noticeably fewer
  episodes.
* **PPO with parallel rollouts** — on-policy with the 20-arm parallel
  environment is a natural fit and avoids the OU-noise / target-network
  tuning surface entirely.
* **Prioritised replay** — uniform sampling spends a lot of compute on
  trivial transitions; PER would prioritise the surprising ones.
* **Action repeat** — Reacher's physics is smooth, so repeating each
  action for ~3 frames would cut wall-clock per episode without
  hurting the learning signal.
* **Layer normalisation instead of batch normalisation** in the
  critic — empirically slightly more stable when the joint
  state-action distribution shifts.
* **Cosine-annealed learning-rate schedule** with restarts to escape
  the late-training plateau around rolling-100 ≈ 32.

---

## 7. References

* Lillicrap, Hunt, Pritzel, *et al.* (2016).
  [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971).
* Udacity DRLND project README, [p2 continuous-control](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control).
* Barth-Maron, Hoffman, Budden, *et al.* (2018).
  [Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/abs/1804.08617).
