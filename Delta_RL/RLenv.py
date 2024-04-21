import gymnasium as gym
from gymnasium import spaces
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np


from stable_baselines3.common.env_checker import check_env


from field_start import field_gen

#speed of the robot
SPEED = 0.5
TOTAL_FIBRES = 30



def calculate_time(robotVect, fibre_start, fibre_end):
        """ 
        A function to calculate the time taken for a move
        inputs
        ---
        robotVect: the current position of the robot coordintes
        fibre_start: The starting position of the fibre
        fibre_end: The end position of the fibre
        
        Output
        ---
        the time taken in seconds
        """        
        for arg in [robotVect, fibre_start, fibre_end]:
            if not isinstance(arg, (list, tuple, np.ndarray)):
                raise TypeError("All input variables must be arrays.")

        
        speed = 1.0 #placeholder value
        distance_vect = np.abs(fibre_start - robotVect) - np.abs(fibre_end-fibre_start)
        distance = np.sqrt(distance_vect[0]**2 + distance_vect[1]**2)
        T = distance/SPEED
        
        return T


class SimpleDeltaEnv(gym.Env):
    """Custom RL environment following the Gym interface. It is the environment used to choose which fibre a positioner should move.

    Args:
        gym (class): OpenAI Gym class
    """
        
    def __init__(self, N_fibres=TOTAL_FIBRES):
        
        super(SimpleDeltaEnv, self).__init__()
        self.N_fibres = N_fibres
        self.action_space = spaces.Discrete(self.N_fibres)
        
        self.fg = super._array
        self.env_vect = None
        
        self.observation_space = spaces.Box(low=0, high=NUM, shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)
        
    def step(self, action):
        
        return self.observation, self.reward, self.done, self.info
    
    def reset(self):
        self.done = False
        fg = field_gen()
        
        self.W = (0, 0) #For now choosing to have the robot always start at zero zero coords
        self.fibre_coords = fg.full_coords_list
        
        
        self.observation = [self.W, self.fibre_coords, self.num_left] 
        return self.observation

        
        
if __name__ == "__main__":
    env = SimpleDeltaEnv()
    check_env(env, warn=True)
    
        