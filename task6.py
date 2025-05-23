import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation
import torch.nn as nn
import torch
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

# Define Environment
env = gym.make('CarRacing-v2', render_mode="rgb_array", continuous=True)
env = GrayScaleObservation(env, keep_dim=True)

# Init Writer for Tensorboard Display
writer = SummaryWriter(log_dir="ppo_logs")

# Convolution Variables
window_size = 5
stride = 2

# Hyperparameters
action_std = 0.1 # Standard deviation for action Normal distribution
epsilon = 0.2 # How much should policy be allowed to change?
gamma = 0.99 # Discount factor for return calculation
epochs = 4
num_episodes = 1000
learning_rate = 1e-4

# Define Model
class ACNN(nn.Module):
	def __init__(self, action_dimensions):
		super().__init__()

		# For Car-Racing, out input layer ouput shape is a 96x96 image with 3 channels

		# Convolution Layers
		# Three channels, we'll use 16, 18, and 32 filter for each conv layer respectively
		# We used 3 convultion layer for deep processing
		# Use ReLU for activation
		self.conv = nn.Sequential(
			#nn.Conv2d(3, 16, kernel_size=window_size, stride=stride),
			nn.Conv2d(1, 16, kernel_size=window_size, stride=stride),
			nn.ReLU(),
			nn.Conv2d(16, 18, kernel_size=window_size, stride=stride),
			nn.ReLU(),
			nn.Conv2d(18, 32, kernel_size=window_size, stride=stride),
			nn.ReLU()
		)

		# Convolution Output Shape is 32 channels, 9x9 image

		# Fully Connected Layers to translate identified shapes in convultion as features
		# Flattened input (32, 9, 9) will have size 32*9*9 = 2592
		# We will reduce the large size 2592 vector to 128 to make computation easier
		self.fc = nn.Sequential(nn.Flatten(), nn.Linear(2592, 512), nn.ReLU())

		self.actor = nn.Linear(512, action_dimensions) # Outputs action in 3 dimensions (steering, acceleration, gas)
		self.critic = nn.Linear(512, 1) # Outputs V(s)

	def forward(self, state):
		state = self.conv(state)
		state = self.fc(state)
		return self.actor(state), self.critic(state)

# Initalize model, use cuda (or "cpu") for device
device = torch.device("cuda")

model = ACNN(3).to(device)

# Optimizer for automatically adjusting parameters
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Start environment and get inital observation

for episode in range(num_episodes):
	print(f"==========\n\nEPISODE {episode}\n\n==========")
	done = False

	obs, _ = env.reset()

	episode_reward = 0.0

	data = {"states": [], "actions": [], "probabilites": [], "values": [], "rewards": []}

	while not done:
		# Preprocess state and run policy
		state = torch.tensor(obs, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0) / 255.0

		with torch.no_grad():
			action_mean, value = model(state)

		# Sample action from fixed-std Normal distribution (discrete math)
		dist = Normal(action_mean, action_std) # Get mean and standard deviation into dist
		action = torch.tanh(dist.sample()) # Sample Normal distribution
		probability = dist.log_prob(action).sum(dim=1) # Get the logarithm of the probability that action is taken given the distribution
		# We need the probability of the action so we can computer policy loss later, and clip it using PPO.

		# Convert action tensor into numpy array for Gymnasium
		steering = action[..., 0]
		gas = torch.clamp(action[..., 1], min=0.0)
		brake = torch.clamp(action[..., 2], min=0.0)

		action_choice = torch.stack([steering, gas, brake], dim=-1)
		action_choice = action_choice.squeeze(0).cpu().numpy()

		# Get new observation and reward after taking action, find out if terminated

		obs, reward, terminated, truncated, _ = env.step(action_choice)
		done = terminated or truncated

		# Store data for optimization
		# Remove batch dimension from output shapes (state and action)
		# We do this to store them as single experiences in the replay buffer (s, a, s')
		data["states"].append(state.squeeze(0))
		data["actions"].append(action.squeeze(0))
		data["values"].append(value.item())
		data["rewards"].append(reward)
		data["probabilites"].append(probability.item())
		episode_reward += reward

	# Before we end the episode, use PPO to optimize the results!

	# First, we're going to calculate return recursively using G_t = r_t + gamma * G_{t+1}

	returns = []
	G = 0
	# Our base case is the last reward we get, so we must look at out rewards backwards to find the return of the episode
	for r in reversed(data["rewards"]):
		G = r + gamma * G
		returns.insert(0, G)

	returns = torch.tensor(returns, dtype=torch.float32, device=device)

	# Next we'll get a tensor for the advantages of each action
	values_tensor = torch.tensor(data["values"], dtype=torch.float32, device=device)

	# Get advantage by getting the difference between the reward we got and the value of the state
	advantages = returns - values_tensor

	# Combine the states by stacking them into one new tensor, along the 0th dimension
	states_tensor = torch.stack(data["states"], 0)
	action_tensor = torch.stack(data["actions"], 0)
	old_prob_tensor = torch.tensor(data["probabilites"], dtype=torch.float32, device=device)

	print(states_tensor)

	for _ in range(epochs):
		new_means, new_values = model(states_tensor)
		dist = Normal(new_means, action_std) # Create a Normal distribition using mean from mean of concatenated tensors

		log_probs = dist.log_prob(action_tensor).sum(dim=1)

		# Let's get a representation of the change in policy, from old to new
		ratio = (log_probs - old_prob_tensor).exp()

		# Finally, we can calculate the surrogate policy loss (see equation in paper)

		# We need to get the clipped and unclipped surrogate objectives
		L_uncp = advantages * ratio

		# Clip surrgoate with torch.clamp() using epsilon hyperparameter
		L_clip = advantages * torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

		ppo_loss = -torch.min(L_uncp, L_clip).mean()
		value_loss = 0.5 * (returns - new_values.squeeze()).pow(2).mean()
		loss = ppo_loss + value_loss

		# Pass loss to optimizer to configure parameters for next ep
		optimizer.zero_grad()
		loss.backward()
		clip_grad_norm_(model.parameters(), max_norm=0.5)
		optimizer.step()

	writer.add_scalar("Reward/Episode", episode_reward, episode)
	writer.add_scalar("Loss/Policy", ppo_loss.item(), episode)
	writer.add_scalar("Loss/Value", value_loss.item(), episode)

writer.close()
env.close()
