#!/usr/bin/env python
"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Modified from the script written by Jonathan Ho (hoj@openai.com)
"""

import os
import argparse
import pickle
import numpy as np
import gym
import load_policy

from tqdm import tqdm


def run_expert(expert_policy_file, envname, max_timesteps, num_rollouts=20, render=False, quiet=False):
    policy_net = load_policy.load_policy(expert_policy_file)

    env = gym.make(envname)
    max_steps = max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in tqdm(range(num_rollouts)):
        if not quiet:
            print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_net(obs[None, :])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0 and not quiet:
                print(f"{steps}/{max_steps}")
            if steps >= max_steps:
                break
        returns.append(totalr)

    if not quiet:
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}

    if not os.path.exists('expert_data'):
        os.makedirs('expert_data')

    with open(os.path.join('expert_data', envname + '.pkl'), 'wb') as f:
        pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

    return returns


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    run_expert(args.expert_policy_file, args.envname, args.max_timesteps, args.num_rollouts, args.render)
