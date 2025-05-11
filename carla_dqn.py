import gym
import torch.nn as nn
import torch
import carla
import gym_carla
import random
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from numpy import mean

params = {
	'number_of_vehicles': 1,
	'number_of_walkers': 0,
	'display_size': 256,  # screen size of bird-eye render
	'max_past_step': 1,  # the number of past steps to draw
	'dt': 0.1,  # time interval between two frames
	'discrete': True,  # whether to use discrete control space
	'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
	'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
	'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
	'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
	'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
	'port': 2000,  # connection port
	'town': 'Town03',  # which town to simulate
	'max_time_episode': 1000,  # maximum timesteps per episode
	'max_waypt': 12,  # maximum number of waypoints
	'obs_range': 32,  # observation range (meter)
	'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
	'd_behind': 12,  # distance behind the ego vehicle (meter)
	'out_lane_thres': 2.0,  # threshold for out of lane
	'desired_speed': 8,  # desired speed (m/s)
	'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
	'display_route': True,  # whether to render the desired route
}

# Define Environment
env = gym.make('carla-v0', params=params)

# Init Writer for Tensorboard Display
writer = SummaryWriter(log_dir="dqn_logs")

# Hyperparameters
num_episodes = 100
epsilon_init = 1.0
epsilon = epsilon_init # Epsilon-greedy search
epsilon_decay_episodes = 30 # After how many episodes do we decay down to the min?
epsilon_min = 0.1 # Minimum ratio of random actions
gamma = 0.8 # Discount factor for return calculation
learning_rate = 1e-4
target_update_frequency = 200 # How often do we update Target Network?
warmup_threshold = 10000 # We need at this many experiences in the buffer to conclude warmup
buffer_capacity = 20000

# We are using a Q-Network on a continuous action space, so we need to discretize the actions!
disc_actions = [
	[0, 1.0, 0], # Full forward
	[0, 0, 0.8], # Brake
	[-1.0, 1.0, 0], # Full left
	[-0.5, 1.0, 0], # Half right
	[1.0, 1.0, 0], # Full right
	[0.5, 1.0, 0], # Half right
	[0, 0, 0], # Do nothing
	[0, 1.0, 0.8], # Slow down
]

# Define Model
class DQN(nn.Module):
	def __init__(self, action_count):
		super().__init__()

		# For Car-Racing, out input layer ouput shape is a 96x96 image with 3 channels

		# Convolution Layers
		# Three channels, we'll use 32, 64, and 64 again to filter for each conv layer respectively
		# We used 3 convultion layer for deep processing
		# Use ReLU for activation
		self.conv = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())

		# Convolution Output Shape is 64 channels, 8x8 feature map

		self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(50176, 6272), nn.ReLU())
		self.fc2 = nn.Sequential(nn.Flatten(), nn.Linear(6272, 784), nn.ReLU())
		self.fc3 = nn.Sequential(nn.Flatten(), nn.Linear(784, action_count))

	def forward(self, state):
		state = self.conv(state)
		state = self.fc1(state)
		state = self.fc2(state)
		state = self.fc3(state)
		return state

def compute_bellman(reward, next_state, gamma, compute_model):
	# Compute Bellman Target

	# We're calling the model but we don't need the gradient - just the return, so no_grad
	with torch.no_grad():
		next_q_values = compute_model(next_state)
		best_next_q, _ = torch.max(next_q_values, dim=1)
		bellman_target = reward + gamma * best_next_q
	return bellman_target

class replay_buffer():
	def __init__(self, capacity):
		self.buffer = []
		self.capacity = capacity

	def sample(self):
		return random.sample(self.buffer, 1)

	def add(self, experience):
		if len(self.buffer) == self.capacity:
			self.buffer.pop(0)

		self.buffer.append(experience)

	def length(self):
		return len(self.buffer)

# Initalize model, use cuda (or "cpu") for device
device = torch.device("cuda")

num_actions = len(params["discrete_acc"]) * len(params["discrete_steer"])
model = DQN(num_actions).to(device)
target_model = DQN(num_actions).to(device)

# Optimizer for automatically adjusting parameters
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

rbuffer = replay_buffer(buffer_capacity)

# Start environment and get inital observation

for episode in range(num_episodes):
	print(f"==========\n\nEPISODE {episode}\n\n==========")
	done = False
	obs = env.reset()
	obs = obs[0]
	episode_reward = 0.0
	data = {"loss": []}
	n = 0
	steps = 0

	while not done:
		# STEP 1: GENERATE EXPERIENCE USING POLICY MODEL
		# Process current state as tensor
		state = torch.tensor(obs, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0) / 255.0

		# Get q_values of each action from the model
		q_values = model(state)

		# Use an epsilon-greedy policy to pick an action
		r = random.random()

		if (r > epsilon):
			action_index = torch.argmax(q_values).item()
		else:
			action_index = random.randint(0, len(disc_actions) - 1)

		# action_choice = disc_actions[action_index]
		action_choice = action_index

		epsilon = max(epsilon_min, epsilon_init - (episode / epsilon_decay_episodes) * (epsilon_init - epsilon_min))

		# Get new experience after taking action, find out if terminated
		obs, reward, terminated, info = env.step(action_choice)
		obs = obs[0]

		next_state = torch.tensor(obs, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0) / 255.0
		done = terminated
		episode_reward += reward

		# STEP 2: STORE EXPERIENCE IN THE REPLAY BUFFER
		rbuffer.add({"state": state, "action": action_choice, "reward": reward, "next": next_state, "done": done})

		# STEP 3: SAMPLE EXPERIENCE BUFFER AND GENERATE LOSS (IF WARM UP COMPLETED)
		if rbuffer.length() > warmup_threshold:
			xp = rbuffer.sample()[0]

			# 3a: GET TARGET Q FROM TARGET NETWORK
			if xp["done"]:
				target_q = torch.tensor(0.0, device=device)
			else:
				# Predict Q Value using Target (Slow Update) Model
				target_q = compute_bellman(torch.tensor([xp["reward"]], device=device), xp["next"],  gamma, target_model).squeeze()

				# Predict Q Value using Policy (Fast Update) Model
				# target_q = compute_bellman(state, torch.tensor([reward], device=device), next_state,  gamma, model).squeeze()

			# 3b: PREDICT Q WITH POLICY NETWORK
			predicted_values = model(xp["state"])
			#predicted_q = predicted_values[0][disc_actions.index(xp["action"])]
			predicted_q = predicted_values[0][xp["action"]]

			# Create loss using mean-squared error (MSE)
			value_loss = torch.nn.functional.mse_loss(predicted_q, target_q)

			# Update model parameters using Adam optimizer
			optimizer.zero_grad()
			value_loss.backward()
			optimizer.step()

			# Log loss for tensorboard
			data["loss"].append(value_loss.item())

		# STEP 4: MAINTAIN TARGET NETWORK
		# Every few steps...
		n = n + 1
		if n >= target_update_frequency:
			print(f"Episode {episode}, Copying policy network to target")
			n = 0
			# Copy weights to Target Q Network, we'll use this for bellman because its more stable than the policy model
			target_model.load_state_dict(model.state_dict())

		steps += 1

	if len(data["loss"]) > 0:
		mean_episode_loss = mean(data["loss"])
	else:
		mean_episode_loss = 0

	if rbuffer.length() < warmup_threshold: print("(WARMING UP) ", end="")
	print(f"Episode {episode} concluded after {steps} steps")
	print(f" - Ending experience buffer size: {rbuffer.length()}")
	print(f" - Ending epsilon value: {epsilon}")
	writer.add_scalar("Reward/Episode", episode_reward, episode)
	writer.add_scalar("Loss/Value", mean_episode_loss, episode)

writer.close()
env.close()
