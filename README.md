# Policy-Based RL — Reacher (DDPG)

Final project for Udacity's *Deep Reinforcement Learning Nanodegree*
(nd893) — **Project 2: Continuous Control**. A from-scratch DDPG agent
that solves Unity ML-Agents' **20-arm Reacher** environment.

## What the project does

* **Environment.** Twenty robotic arms in parallel; each arm has a
  hand that earns +0.1 reward per timestep it stays inside a moving
  target sphere.
* **Goal.** Drive the rolling-100-episode mean of the per-episode
  per-arm score above **+30**.
* **Approach.** A single DDPG agent (one Actor + one Critic, both
  with target networks) that learns from the joint experience of all
  20 arms. Every environment step adds 20 transitions to a shared
  replay buffer.

For the full algorithm + hyperparameter rationale + future work,
see [`Report.md`](Report.md).

## Files

```
model.py                    Actor + Critic networks (both with BatchNorm on the first layer)
agent.py                    DDPGAgent + ReplayBuffer + OUNoise + soft-update logic
train.py                    Standalone training script (CLI)
Continuous_Control.ipynb    Jupyter notebook used for training (mirrors train.py)
Report.md                   Algorithm, hyperparameters, results, future work
README.md                   You are here
reacher_actor.pth           Trained Actor weights         (produced by training)
reacher_critic.pth          Trained Critic weights        (produced by training)
reacher_scores.npy          Per-episode mean-score curve  (produced by training)
reacher_training_curve.png  Rendered learning curve       (produced by training)
```

## Installation

Python 3.10 or 3.11 is recommended (the Udacity `unityagents` shim
hasn't been touched since 2018 and is best matched to Python ≤ 3.11).

```bash
# 1. Core ML deps
pip install torch numpy matplotlib

# 2. Udacity's unityagents shim, plus its old protobuf pin.
#    The shim ships in the project workspace under home/python/. After
#    cloning that, run:
#       pip install -e ./home/python
pip install 'protobuf<3.21'

# 3. Download the **20-agent macOS** Unity Reacher binary
#    (the canonical Udacity URL):
#    https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip
#    (or pick the Linux/Windows build if you're not on macOS — see the
#    upstream nd893 README at p2_continuous-control)
unzip Reacher.app.zip          # produces Reacher.app/
```

If you're on **Apple Silicon** (M-series) the macOS Unity build is
x86_64. It runs under Rosetta 2 — install it with
`softwareupdate --install-rosetta` if it isn't already there.

The Unity package's protobuf files were generated against an older
protobuf API, so you may need to set the env var
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` when invoking Python.

## Running the training

### Option A — CLI

```bash
python3 train.py --env path/to/Reacher.app --episodes 250
```

### Option B — Jupyter notebook

Open [`Continuous_Control.ipynb`](Continuous_Control.ipynb) and run
all cells. The notebook walks through the env spec, then runs the
same DDPG training loop and writes the same checkpoints.

The agent uses **MPS** (Apple Silicon GPU) automatically when
available, falling back to CUDA, then CPU. The training loop saves
`reacher_actor.pth`, `reacher_critic.pth`, `reacher_scores.npy`, and
the rendered learning curve once the rolling-100-episode mean exceeds
+30.

## Expected results

Typical solve time with the hyperparameters in `agent.py`:

* Rolling-100 mean **crosses +30 between episodes 150 and 200**
* Final per-episode score plateaus around **31–33**
* Per-episode wall-clock time on Apple Silicon (M3 Pro, MPS): about
  **10–15 seconds**, so the full run is roughly 30–50 minutes

If your run stalls below +25 by episode 150, the most common causes
are documented in `Report.md` § 5 — usually it's BatchNorm, the
update cadence, or the OU-noise reset.

## License

Educational submission for Udacity nd893. Unity ML-Agents Reacher
environment © Unity / Udacity.
