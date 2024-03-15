from gym.envs.registration import register

register(
     id='Delta-v1',
     entry_point='Delta_RL.gymEnv.customEnv.envs:SimpleDeltaEnv',
 )