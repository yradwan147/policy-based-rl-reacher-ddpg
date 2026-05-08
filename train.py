"""Train DDPG on the Unity Reacher (multi-agent) environment."""
import argparse
from collections import deque

import numpy as np
import torch
from unityagents import UnityEnvironment

from agent import DDPGAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--target", type=float, default=30.0)
    ap.add_argument("--save", default="reacher")
    args = ap.parse_args()

    env = UnityEnvironment(file_name=args.env)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    info = env.reset(train_mode=True)[brain_name]
    num_agents = len(info.agents)
    state_size = info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size
    print(f"Reacher env: {num_agents} agents, state={state_size}, action={action_size}")

    agent = DDPGAgent(state_size=state_size, action_size=action_size,
                      num_agents=num_agents)
    scores_window = deque(maxlen=100)
    all_scores = []

    for ep in range(1, args.episodes + 1):
        info = env.reset(train_mode=True)[brain_name]
        states = info.vector_observations
        agent.reset()
        scores = np.zeros(num_agents)
        while True:
            actions = agent.act(states)
            info = env.step(actions)[brain_name]
            ns = info.vector_observations
            r  = np.array(info.rewards)
            d  = np.array(info.local_done, dtype=int)
            agent.step(states, actions, r, ns, d)
            states = ns
            scores += r
            if d.any():
                break
        score = float(np.mean(scores))
        scores_window.append(score)
        all_scores.append(score)
        avg = np.mean(scores_window)
        print(f"Ep {ep:3d}\tscore={score:.2f}\trolling100={avg:.2f}", end="\r")
        if ep % 10 == 0:
            print(f"\nEpisode {ep}: rolling-100 mean = {avg:.2f}")
        if avg >= args.target and len(scores_window) == 100:
            print(f"\nSolved in {ep} episodes — saving checkpoints")
            torch.save(agent.actor_local.state_dict(),  f"{args.save}_actor.pth")
            torch.save(agent.critic_local.state_dict(), f"{args.save}_critic.pth")
            break

    env.close()
    np.save(f"{args.save}_scores.npy", np.array(all_scores))


if __name__ == "__main__":
    main()
