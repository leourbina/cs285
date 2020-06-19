#!/usr/bin/env python
"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Modified from the script written by Jonathan Ho (hoj@openai.com)
"""

import argparse
import load_policy
from agent import run_agent


def run_expert(expert_policy_file, config):
    policy_net = load_policy.load_policy(expert_policy_file)

    return run_agent(policy_net, config)


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
