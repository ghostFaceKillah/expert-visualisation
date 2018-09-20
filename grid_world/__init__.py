from gym.envs.registration import register

register(
    id='grid_world-v0',
    entry_point='grid_world.grid_world_env:GridWorldEnv'
)