import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.env_checker import check_env

#speed of the robot
SPEED = 0.5
class Delta(gym.Env):
    """Custom RL environment following the Gym interface. It is the environment used to choose which fibre a positioner should move.

    Args:
        gym (class): OpenAI Gym class
    """
    
    
    def __init__(self) -> None:
        super(Delta, self).__init__()
        
        
    def calculate_time(self, robotVect, fibre_start, fibre_end):
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
        
        
        
        
if __name__ == "__main__":
    env = Delta()
    check_env(env, warn=True)
    
        