import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation
import torch.nn as nn
import torch
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_

device = torch.device("cuda")
writer = SummaryWriter(log_dir="training_logs")
num_action_dimensions = 3
num_episodes = 1500

class ACNN2(nn.Module):
	def __init__(self, action_dimensions):
		super().__init__()

		window_size = 5
		stride = 2

		self.conv = nn.Sequential(	
			# 1, 96, 96
			nn.Conv2d(1, 16, kernel_size=window_size, stride=stride), # 16, 46, 46
			nn.ReLU(),
			nn.Conv2d(16, 18, kernel_size=window_size, stride=stride), # 18, 21, 21
			nn.ReLU(),
			nn.Conv2d(18, 32, kernel_size=window_size, stride=stride), # 32, 9, 9
			nn.ReLU())

		self.fc = nn.Sequential(
			nn.Flatten(), 
			nn.Linear(2592, 1296), 
			nn.ReLU(), 
			nn.Linear(1296, 512), 
			nn.ReLU()
		)

		self.actor = nn.Linear(512, action_dimensions)
		self.critic = nn.Linear(512, 1)

	def forward(self, state):
		state = self.conv(state)
		state = self.fc(state)
		return self.actor(state), self.critic(state)

class Config:
	def __init__( self,
		name,
		action_std_init,
		action_std_min,
		warmup_episodes,
		decay_episodes,
		epsilon,
		gamma, 
		epochs,
		learning_rate,
		clip_max_norm,
		nn_weight_clamp
	):
		self.name = name
		self.action_std_init = action_std_init
		self.action_std_min = action_std_min
		self.warmup_episodes = warmup_episodes
		self.decay_episodes = decay_episodes
		self.epsilon = epsilon
		self.gamma = gamma
		self.epochs = epochs
		self.learning_rate =  learning_rate
		self.clip_max_norm = clip_max_norm
		self.nn_weight_clamp = nn_weight_clamp

		self.env = GrayScaleObservation(
			gym.make(
				'CarRacing-v2', 
				render_mode="rgb_array", 
				continuous=True
			), 
			keep_dim=True
		)

		self.model = ACNN2(num_action_dimensions).to(device)
		self.optimizer = torch.optim.Adam(
			self.model.parameters(), 
			lr=self.learning_rate
		)

		self.action_std = self.action_std_init

		self.episode = 0
		self.step = 0


configurations = [
	Config (
		name = "PPO_1",
		action_std_init = 0.7,
		action_std_min = 0.1,
		warmup_episodes = 100,
		decay_episodes = 150,
		epsilon = 0.4,
		gamma = 0.8,
		epochs = 4,
		learning_rate = 1e-7, 
		clip_max_norm = 0.5,
		nn_weight_clamp = 1e3,
	),
	Config (
		name = "PPO_2",
		action_std_init = 0.7,
		action_std_min = 0.1,
		warmup_episodes = 100,
		decay_episodes = 150,
		epsilon = 0.5,
		gamma = 0.8,
		epochs = 4,
		learning_rate = 1e-7, 
		clip_max_norm = 0.5,
		nn_weight_clamp = 1e3,
	),
	Config (
		name = "PPO_3",
		action_std_init = 0.7,
		action_std_min = 0.1,
		warmup_episodes = 100,
		decay_episodes = 150,
		epsilon = 0.3,
		gamma = 0.8,
		epochs = 4,
		learning_rate = 1e-7, 
		clip_max_norm = 0.5,
		nn_weight_clamp = 1e3,
	),
	
	
]

if __name__ == "__main__":
	for episode in range(num_episodes):

		log_data = [{
			'name': x.name,
			'reward': 0,
			'v_loss': 0,
			'p_loss': 0
		} for x in configurations]
		
		log_index = 0

		for params in configurations:
			print(f"Episode {episode + 1}.{configurations.index(params) + 1} begin.")
			done = False

			obs, _ = params.env.reset()

			episode_reward = 0.0

			data = {"states": [], "actions": [], "probabilites": [], "values": [], "rewards": []}

			while not done:
				# Preprocess state and run policy
				state = torch.tensor(obs, dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0) / 255.0

				with torch.no_grad():
					action_mean, value = params.model(state)

				# Sample action from fixed-std Normal distribution (discrete math)
				dist = Normal(action_mean, params.action_std) # Get mean and standard deviation into dist
				action = torch.tanh(dist.sample()) # Sample Normal distribution
				probability = dist.log_prob(action).sum(dim=1) # Get the logarithm of the probability that action is taken given the distribution
				# We need the probability of the action so we can computer policy loss later, and clip it using PPO.

				# Convert action tensor into numpy array for Gymnasium
				steering = torch.clamp(action[..., 0], min=-1.0, max=1.0)
				gas = torch.clamp(action[..., 1], min=0.0)
				brake = torch.clamp(action[..., 2], min=0.0)

				action_choice = torch.stack([steering, gas, brake], dim=-1)
				action_choice = action_choice.squeeze(0).cpu().numpy()

				# Get new observation and reward after taking action, find out if terminated

				obs, reward, terminated, truncated, _ = params.env.step(action_choice)
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
				G = r + params.gamma * G
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

			for _ in range(params.epochs):
				new_means, new_values = params.model(states_tensor)
				dist = Normal(new_means, params.action_std) # Create a Normal distribition using mean from mean of concatenated tensors

				log_probs = dist.log_prob(action_tensor).sum(dim=1)

				# Let's get a representation of the change in policy, from old to new
				ratio = (log_probs - old_prob_tensor).exp()

				# Finally, we can calculate the surrogate policy loss (see equation in paper)

				# We need to get the clipped and unclipped surrogate objectives
				L_uncp = advantages * ratio

				# Clip surrgoate with torch.clamp() using epsilon hyperparameter
				L_clip = advantages * torch.clamp(ratio, 1 - params.epsilon, 1 + params.epsilon)

				ppo_loss = -torch.min(L_uncp, L_clip).mean()
				value_loss = 0.5 * (returns - new_values.squeeze()).pow(2).mean()
				loss = ppo_loss + value_loss

				# Pass loss to optimizer to configure parameters for next ep
				params.optimizer.zero_grad()
				loss.backward()
				clip_grad_norm_(params.model.parameters(), max_norm=params.clip_max_norm)
				params.optimizer.step()

				with torch.no_grad():
					for p in params.model.parameters():
						p.data = p.data.clamp(-params.nn_weight_clamp, params.nn_weight_clamp)

			print(f"Episode {episode + 1}.{configurations.index(params) + 1} concluded.")
			print(f" - Ending standard deviation: {params.action_std}")
			print(f" - Ending reward {episode_reward}")
			print(f" - Ending loss {value_loss.item()}")

			if (episode > params.warmup_episodes):
				decay_ratio = (episode - params.warmup_episodes) / params.decay_episodes
				params.action_std = max(params.action_std_min, params.action_std_init - decay_ratio * (params.action_std_init - params.action_std_min))

			log_data[log_index]['reward'] = episode_reward
			log_data[log_index]['v_loss'] = value_loss.item()
			log_data[log_index]['p_loss'] = ppo_loss.item()
 	
			log_index += 1

		writer.add_scalars("Reward/Episode", {
			log_data[x]['name']: log_data[x]['reward'] for x in range(len(log_data))
		}, episode)
		writer.add_scalars("Loss/Policy", {
			log_data[x]['name']: log_data[x]['p_loss'] for x in range(len(log_data))
		}, episode)
		writer.add_scalars("Loss/Value", {
			log_data[x]['name']: log_data[x]['v_loss'] for x in range(len(log_data))
		}, episode)
