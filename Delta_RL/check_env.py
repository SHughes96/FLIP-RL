# import gym
from stable_baselines3.common.env_checker import check_env
from RLenv import SimpleDeltaEnv


env = SimpleDeltaEnv(30)

check_env(env)

episodes = 50

for episode in range(episodes):
    done = False
    obs = env.reset()
    while True:
        random_action = env.action_space.sample()
        print('random action ', random_action)
        obs, reward, done, _, info = env.step(random_action)
        print('obs', obs)
        # print('reward', reward)
        #print('info', info)