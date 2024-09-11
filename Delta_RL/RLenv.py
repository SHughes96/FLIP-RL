import gymnasium
from gymnasium import spaces
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

import numpy as np


# from stable_baselines3.common.env_checker import check_env


from field_start import field_gen

#speed of the robot
SPEED = 0.5
TOTAL_FIBRES = 30

def check_if_done(Nfib, array):
    #Assume if is finished for now
    done = True
    #assert len(array)==Nfib*4, 'flattened coords array must be twice the length of the number of fibres available'
    try:
        array = array.reshape(Nfib*2, 2)
    except Exception as e:
        print(f"Failed to reshape array")
        
    for i in range(0, Nfib*2, 2):
        start_coords, end_coords = array[i], array[i+1]
        done = np.array_equal(start_coords, end_coords)
        if not done:
            return False 
    return True


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


class SimpleDeltaEnv(gymnasium.Env):
    """Custom RL environment following the Gym interface. It is the environment used to choose which fibre a positioner should move.

    Args:
        gym (class): OpenAI Gym class
    """
        
    def __init__(self, N_fibres=TOTAL_FIBRES, W=np.array([0,0])):
        
        super(SimpleDeltaEnv, self).__init__()
        self.done = False
        self.N_fibres = N_fibres
        self.action_space = spaces.Discrete(self.N_fibres)
        self.field = field_gen(self.N_fibres)
        
        self.observation_space = spaces.Box(low=0, high=2.*np.pi, shape=(4*self.N_fibres + 2,), dtype=np.float64)
        
        self.W = np.asarray(W)
        self.fibre_coords = self.field.flattened_coords_radial
        self.observation = np.concatenate((self.W.ravel(), self.field.flattened_coords_radial))
        return
        
    def step(self, action):
        
        if action < 0 or action >= self.N_fibres:
            raise ValueError("Invalid action. Action must be in the range [0, {}), but got {}.".format(self.N_fibres, action))
        
        self.fibre_coords = self.fibre_coords.reshape(self.N_fibres*2, 2)
        
        current_index = action * 2
        end_index = current_index + 1
        
        current_coords = self.fibre_coords[current_index]
        end_coords = self.fibre_coords[end_index]
        
        timetaken = calculate_time(self.W, current_coords, end_coords)
        self.reward = -timetaken
        
        #after reward is found, set the current coords and robot coords to the end_coords
        self.fibre_coords[current_index] = self.fibre_coords[end_index]
        self.W = self.fibre_coords[end_index]
        #reflatten the array
        self.fibre_coords = self.fibre_coords.flatten()
        
        self.done = check_if_done(self.N_fibres, self.fibre_coords)
        if self.done:
            self.reward += 100
    
        self.observation = np.concatenate((self.W, self.fibre_coords))
        return self.observation, self.reward, self.done, False, self.info
    
    def reset(self, seed=None):
        self.done = False
        
        self.field = field_gen(self.N_fibres)
        
        self.W = np.asarray((0, 0)) #For now choosing to have the robot always start at zero zero coords
      
        self.fibre_coords = self.field.flattened_coords_radial
        
        self.info = {}
        self.obs = np.concatenate((self.W.ravel(), self.fibre_coords))
        return self.obs, self.info

        
# if __name__ == "__main__":
#     env = SimpleDeltaEnv()
#     check_env(env, warn=True)