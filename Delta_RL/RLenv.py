import gym
from gymnasium import spaces
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import gymnasium

import numpy as np
import copy


# from stable_baselines3.common.env_checker import check_env


from field_start import field_gen

#speed of the robot


def check_if_done(Nfib, array):
    
    array = array.reshape(Nfib, 4)
    done = np.allclose(array[:, :2], array[:, 2:])
    return done


def calculate_time(SPEED, robotVect, fibre_start, fibre_end):
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
            
        x_start = fibre_start[0]*np.cos(fibre_start[1])
        y_start = fibre_start[0]*np.sin(fibre_start[1])
        
        x_end = fibre_end[0]*np.cos(fibre_end[1])
        y_end = fibre_end[0]*np.sin(fibre_end[1])
        
        start = np.array([x_start, y_start])
        end = np.array([x_end, y_end])

        
        #distance = np.linalg.norm(fibre_start - robotVect) + np.linalg.norm(fibre_end - fibre_start)
        distance = np.linalg.norm(start - robotVect) + np.linalg.norm(end - fibre_start)
        #distance = np.sqrt(distance_vect[0]**2 + distance_vect[1]**2)
        T = distance/SPEED
        
        return T, distance
    



class SimpleDeltaEnv(gymnasium.Env):

    """Custom RL environment following the Gym interface. It is the environment used to choose which fibre a positioner should move.

    Args:
        gym (class): OpenAI Gym class
    """
        
    def __init__(self, N_fibres=3, W=np.array([0,0]), SPEED=1.0, REPEAT_PEN=10):
        
        self.SPEED = SPEED
        #
        self.REPEAT_PEN = REPEAT_PEN
        
        super(SimpleDeltaEnv, self).__init__()
        self.done = False
        self.N_fibres = N_fibres
        self.action_space = spaces.Discrete(self.N_fibres)
        self.field = field_gen(self.N_fibres)
        
        self.episode_length = 0
        
        self.observation_space = spaces.Box(low=0, high=2.*np.pi, shape=(4*self.N_fibres + 2,), dtype=np.float64)
        
        self.W = np.asarray(W)
        self.fibre_coords = self.field.flattened_coords_radial
        self.observation = np.concatenate((self.W.ravel(), self.field.flattened_coords_radial))
        self.minimum_distance = self.field.minimum_distance
        
        return
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj

        # Deepcopy the attributes
        for attr, value in self.__dict__.items():
            setattr(new_obj, attr, copy.deepcopy(value, memo))

        return new_obj
    
        
    def step(self, action):
        self.episode_length += 1
        self.info = dict()
        
        if action < 0 or action >= self.N_fibres:
            raise ValueError("Invalid action. Action must be in the range [0, {}), but got {}.".format(self.N_fibres, action))
        
        self.fibre_coords = self.fibre_coords.reshape(self.N_fibres*2, 2)
        
        current_index = action * 2
        end_index = current_index + 1
        
        current_coords = self.fibre_coords[current_index]
        end_coords = self.fibre_coords[end_index]
        
        timetaken, dist = calculate_time(self.SPEED, self.W, current_coords, end_coords)
        self.reward = -timetaken
        if timetaken <= 1e-4:
            self.reward -= self.REPEAT_PEN
            
        if np.allclose(current_coords, end_coords):
            self.reward -= self.REPEAT_PEN
        
        self.info['DISTANCE'] = dist
        
        #after reward is found, set the current coords and robot coords to the end_coords
        self.fibre_coords[current_index] = self.fibre_coords[end_index]
        
        self.W = self.fibre_coords[end_index]
        #reflatten the array
        self.fibre_coords = self.fibre_coords.flatten()
        
        
        self.done = check_if_done(self.N_fibres, self.fibre_coords)
        if self.done:
            self.reward += 100
            self.info['episode'] = {'l':self.episode_length, 'r': self.reward}
    
        self.observation = np.concatenate((self.W, self.fibre_coords))
        #return self.observation, self.reward, self.done, False, self.info
        return (self.observation, self.reward, self.done, False, self.info)
    
    def reset(self, seed=None):
        self.done = False
        self.episode_length = 0
        
        self.field = field_gen(self.N_fibres)
        
        self.W = np.asarray((0, 0)) #For now choosing to have the robot always start at zero zero coords
      
        self.fibre_coords = self.field.flattened_coords_radial
        
        self.info = {}
        self.obs = np.concatenate((self.W.ravel(), self.fibre_coords))
        return (self.obs, self.info)

        
if __name__ == "__main__":
    env = SimpleDeltaEnv()
    #check_env(env, warn=True)
    # Test the environment with random actions
    obs, info = env.reset()
    print(f"Initial observation: {obs}")

    for step in range(100):  # Take 10 steps as a simple test
        action = env.action_space.sample()  # Take random action
        obs, reward, done, _, info = env.step(action)
        print(f"Step {step+1} - Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            print("Episode finished!")
            break