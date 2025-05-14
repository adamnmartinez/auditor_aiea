# This file is modified from <https://github.com/cjy1992/gym-carla.git>:
# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import gym
import gym_carla
import os
import carla
from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.schedules import LinearSchedule

def main():
  # parameters for the gym_carla environment
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

  model_path = "dqn_weights.zip"

  # Set gym-carla environment
  env = gym.make('carla-v0', params=params)

  print("\n=======\n\nChecking for presaved model...\n\n=======\n")

  if os.path.exists(model_path):
    print("\n=======\n\nModel Found...\n\n=======\n")
    model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./dqn_training_logs", exploration_fraction=0.2, exploration_final_eps=0.1)
    model = DQN.load(model_path, env=env)
  else:
    print("\n=======\n\nNo Model Found... Quitting...\n\n=======\n")
    exit()

  def evaluate(model, env, num_episodes=10):
    obs = env.reset()
    episode_rewards = []

    for ep in range(num_episodes):
      done = False
      total_rewards = 0
      steps = 0
      while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        total_rewards += rewards
      episode_rewards.append(total_rewards)
      print(f"EVAL {ep} DONE")

    return episode_rewards

  print(f"\n=======\n\nEvaluating Model...\n\n=======\n")

  episode_rewards = evaluate(model, env, 10)
  avg_reward = sum(episode_rewards) / len(episode_rewards)

  print(f"\n=======\n\nEvaluation Complete\n\n=======\n")
  print(episode_rewards)
  print(f" - AVG: {avg_reward}")
  print("\n==============\n")

if __name__ == '__main__':
  main()
