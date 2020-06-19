import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def agent_wrapper(agent, config):
  def _no_grad_agent(obs):
    with torch.no_grad():
      obs = torch.from_numpy(obs).float().to(config.device)
      return agent(obs).detach().cpu().numpy()

  return _no_grad_agent


class Agent(nn.Module):
  """
  Simple feed forward agent

  Arguments:
      input_size: Size of the input
      output_size: Size of the output
      hidden_layers: Array with sizes of hidden layers
  """

  def __init__(self, input_size, output_size, hidden_layers):
    super().__init__()

    self.input_size = input_size
    self.output_size = output_size
    self.hidden_layers = hidden_layers

    hidden_layers = [input_size] + hidden_layers + [output_size]
    layers = []
    for in_features, out_features in zip(hidden_layers, hidden_layers[1:]):
      layers.append(nn.Linear(in_features, out_features))
      layers.append(nn.ReLU())

    # Pop out the last ReLU
    layers.pop()

    self.layers = nn.Sequential(*layers)

  def forward(self, x):
    x = torch.Tensor(x)
    return self.layers(x)


def run_agent(agent, config):
    max_steps = config.max_timesteps or config.env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in tqdm(range(config.num_rollouts), desc="Rollouts", leave=True):
        if not config.quiet:
            print('iter', i)
        obs = config.env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = agent(obs[None, :])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = config.env.step(action)
            totalr += r
            steps += 1
            if config.render:
                config.env.render()
            if steps % 100 == 0 and not config.quiet:
                print(f"{steps}/{max_steps}")
            if steps >= max_steps:
                break
        returns.append(totalr)

    if not config.quiet:
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

    return actions, observations, returns
