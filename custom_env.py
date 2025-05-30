import gymnasium as gym
fromm gymnasium import Wrapper
from gymnasium.wrappers import GrayScaleObservation

class CustomWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
