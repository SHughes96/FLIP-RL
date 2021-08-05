import gym
from gym import error, spaces, utils
from gym.utils import seeding


class SimpleDeltaEnv(gym.Env):

    def __init__(self) -> None:
        super().__init__()
        return