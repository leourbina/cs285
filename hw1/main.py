"""
Runs trained agents

Usage:
  main.py <agent_file> <env_name> [--render] [--max-steps=<timesteps>] [--num-rollouts=<rollouts>]

Options:
  -h --help                        Show this message
  --render                         Render mujoco env [default: True]
  --max-steps=<max_timesteps>  Max number of steps [default: 1000]
  --num-rollouts=<num_rollouts>    Number of rollouts [default: 10]
"""
from docopt import docopt

import gym
import torch
import numpy as np

from agent import Agent


def main(env_name, agent_file, num_rollouts=20, max_steps=1000, render=False):
  print(env_name, agent_file, num_rollouts, max_steps, render)
  state_dict, n_inputs, n_outputs, hidden_layers = torch.load(f"{agent_file}")
  agent = Agent(n_inputs, n_outputs, hidden_layers)
  agent.load_state_dict(state_dict)

  env = gym.make(env_name)

  returns = []

  print("num_rollouts", num_rollouts)
  for i in range(num_rollouts):
    print('iter', i)
    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0

    while not done:
      action = agent(obs).unsqueeze(0).detach().cpu().numpy()
      obs, r, done, _ = env.step(action)
      totalr += r
      steps += 1

      if render:
        env.render()

      if steps % 100 == 0:
        print(f"{steps}/max_steps")

      if steps >= max_steps:
        break

    returns.append(totalr)

  print('returns', returns)
  print('mean return', np.mean(returns))
  print('std of return', np.std(returns))


if __name__ == '__main__':
  args = docopt(__doc__)
  agent_file = args["<agent_file>"]
  env_name = args["<env_name>"]
  render = bool(args["--render"])
  max_steps = int(args["--max-steps"])
  rollouts = int(args["--num-rollouts"])

  main(env_name, agent_file, render=render, max_steps=max_steps, num_rollouts=rollouts)
