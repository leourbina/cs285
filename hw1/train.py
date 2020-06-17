import pickle
from tqdm import tqdm

import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F

from agent import Agent

AGENT_DIR = "./trained_agents"


def train_bc_agent(expert, epochs=10, lr=5e-5, save=True):
  expert_data = ExpertData(expert)

  split = 0.9
  N = len(expert_data)

  train_length = int(split * N)
  test_length = len(expert_data) - train_length

  trainset, valset = data.random_split(expert_data, [train_length, test_length])
  train_loader = data.DataLoader(trainset, batch_size=128, shuffle=True)
  test_loader = data.DataLoader(valset, batch_size=128)

  x, y = trainset[0]

  agent = Agent(input_size=x.shape[0], output_size=y.shape[0], hidden_layers=[50, 50])

  train_losses, test_losses = train_epochs(agent, train_loader, test_loader, epochs=epochs, lr=lr)

  if save:
    torch.save((agent.state_dict(), agent.input_size, agent.output_size, agent.hidden_layers), f"{AGENT_DIR}/{expert}-bc")

  return agent, train_losses, test_losses


class ExpertData(data.Dataset):
  def __init__(self, expert):
    with open(f"./expert_data/{expert}.pkl", 'rb') as data:
      expert_data = pickle.load(data)

    self.observations = expert_data['observations']
    self.actions = expert_data['actions']

  def __len__(self):
    return self.observations.shape[0]

  def __getitem__(self, index):
    x = torch.Tensor(self.observations[index]).contiguous()
    y = torch.Tensor(self.actions[index])

    return x, y.squeeze(0)


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
