import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.env_checker import check_env

class Delta(gym.Env):
    """Custom RL environment following the Gym interface. It is the environment used to choose which fibre a positioner should move.

    Args:
        gym (class): OpenAI Gym class
    """
    
    
    def __init__(self) -> None:
        super(Delta, self).__init__()
        
        
        
        
if __name__ == "__main__":
    env = Delta()
    check_env(env, warn=True)
    
        