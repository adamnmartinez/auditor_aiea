import gymnasium as gym
from stable_baselines3 import A2C

# Define Environment
env = gym.make('CarRacing-v2', render_mode="rgb_array", continuous=True)

# Define Model
model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
model.learn(total_timesteps=10_000)


vec_env = model.get_env()

obs = vec_env.reset()

for _ in range(1000):
	# Model Predict Action
	action, state = model.predict(obs, deterministic=True)

	# Take action, get observation, reward, isDone, and info.
	obs, reward, done, info = vec_env.step(action)

	vec_env.render("rgb_array")

###
#episode_finished = False

#while not episode_finished:
#	action = env.action_space.sample()
#	observation, reward, terminated, truncated, info = env.step(action)

#	env.render()

#	print(action)

#	episode_finished = terminated or truncated

#env.close()
###
