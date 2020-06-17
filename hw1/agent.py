import gym
import torch
import torch.nn as nn

from tqdm import tqdm

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


def run_agent(agent_file, envname, max_timesteps, num_rollouts=1, render=False, quiet=False):
  state_dict, n_inputs, n_outputs, hidden_layers = torch.load(agent_file)
  agent = Agent(n_inputs, n_outputs, hidden_layers)
  agent.load_state_dict(state_dict)

  env = gym.make(envname)
  max_steps = max_timesteps or env.spec.timestep_limit

  returns = []
  observations = []

  for i in tqdm(range(num_rollouts)):
    if not quiet:
      print('iter', i)

    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0

    while not done:
      action = agent(obs).detach().cpu().numpy()
      observations.append(obs)

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

  return returns
