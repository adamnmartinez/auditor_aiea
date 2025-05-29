import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation
import torch.nn as nn
import torch
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from numpy import mean
import random

from task6 import run_episode as ppo_episode
from task7 import run_episode as dqn_episode
from task8 import run_episode as ppo2_episode

num_episodes = 1000

class Agent:
	def __init__(self, name, train_func):
		self.name = name
		self.train_func = train_func

	def run_episode(self, writer):
		return self.train_func(self.name, writer)

agents = [
	# Agent("PPO", ppo_episode),
	# Agent("DQN", dqn_episode),
	Agent("PPO+", ppo2_episode)
]

writer = SummaryWriter(log_dir="training_logs")

for episode in range(num_episodes):
	print(f"==========[ Episode {episode + 1}/{num_episodes}, Started ]==========")
	for agent in agents:
		print(f" - Running {agent.name}...")
  #
		# try:
		# 	agent.run_episode(writer)
		# except:
		# 	print("  - An error occured during this run.")

		agent.run_episode(writer)
		print("   - Concluded.")

writer.close()
for agent in agents:
	agent.env.close()
