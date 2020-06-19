from tqdm import tqdm

import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F

from agent import Agent, run_agent, agent_wrapper
from run_expert import run_expert
from load_policy import load_policy


def train_bc_agent(expert_policy_file, config):
  # Run expert
  actions, observations, returns = run_expert(expert_policy_file, config)

  observations = torch.Tensor(observations).to(config.device)
  actions = torch.Tensor(actions).squeeze(1).to(config.device)

  expert_data = data.TensorDataset(observations, actions)

  N = len(expert_data)

  train_length = int(config.split * N)
  test_length = N - train_length

  trainset, valset = data.random_split(expert_data, [train_length, test_length])
  train_loader = data.DataLoader(trainset, batch_size=128, shuffle=True)
  test_loader = data.DataLoader(valset, batch_size=128)

  x, y = expert_data[0]

  agent = Agent(input_size=config.env.observation_space.shape[0], output_size=config.env.action_space.shape[0], hidden_layers=config.hidden_layers)

  train_losses, test_losses = train_epochs(agent, train_loader, test_loader, epochs=config.epochs, lr=config.lr)

  return agent, train_losses, test_losses, expert_data


def train_dagger_agent(expert_policy_file, config):
  # Get the actions that the expert would do given the observations
  expert = load_policy(expert_policy_file)

  agent, train_losses, test_losses, dataset = train_bc_agent(expert_policy_file, config)

  for i in range(config.num_dagger_iterations):
    # Get observations using the policy we just trained
    _, observations, _ = run_agent(agent_wrapper(agent, config), config)
    actions = [expert(obs[None, :]) for obs in observations]

    # Create a new training set that uses all the data
    observations = torch.Tensor(observations).to(config.device)
    actions = torch.Tensor(actions).squeeze(1).to(config.device)

    dagger_data = data.TensorDataset(observations, actions)
    dataset = data.ConcatDataset([dataset, dagger_data])

    N = len(dataset)
    train_length = int(config.split * N)
    test_length = N - train_length

    trainset, valset = data.random_split(dataset, [train_length, test_length])

    train_loader = data.DataLoader(trainset, batch_size=128, shuffle=True)
    test_loader = data.DataLoader(valset, batch_size=128)

    train_loss, test_loss = train_epochs(agent, train_loader, test_loader, epochs=config.epochs, lr=config.lr)

    train_losses += train_loss
    test_losses += test_loss

  return agent, train_losses, test_losses


def train_epochs(model, train_loader, test_loader, epochs=40, optimizer=None, lr=1e-4, loss_func=F.mse_loss):
  """
  Trains model using the given train and test data loaders.

  Returns tuple (train_losses, test_losses)
  """

  if optimizer is None:
    optimizer = optim.Adam(model.parameters(), lr=lr)

  train_losses, test_losses = [], []

  test_loss = eval_loss(model, test_loader, loss_func)
  test_losses.append(test_loss)

  for epoch in tqdm(range(epochs), desc="Epochs", leave=True):

    train_loss = train(model, train_loader, optimizer, loss_func)
    train_losses += train_loss

    test_loss = eval_loss(model, test_loader, loss_func)
    test_losses.append(test_loss)

  return train_losses, test_losses


def train(model, train_loader, optimizer, loss_func):
  model.train()

  losses = []
  for i, (x, target) in enumerate(train_loader):
    y = model(x)
    loss = loss_func(y, target)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return losses


def eval_loss(model, data_loader, loss_func):
  model.eval()
  loss = 0.

  with torch.no_grad():
    for x, target in data_loader:
      y = model(x)
      loss = loss_func(y, target)

      loss += loss * x.shape[0]

    loss /= len(data_loader.dataset)

  return loss.item()
