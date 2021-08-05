from gym.envs.registration import register

register(
    id='DeltaEnvSimple-v0',
    entry_point='Delta_RL.envs:SimpleDeltaEnv',
)
