import grid_world

import gym

"""
First we will start with baselines to remove the possibility of 
getting the implementation wrong.
"""


def make_vec_env(env_id, num_env, seed):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    def make_env(rank):
        def _thunk():
            env =  gym.make(env_id)
            env.seed(seed + rank)
            return env
        return _thunk

    # set_global_seeds(seed)
    return vec_env.subproc_vec_env.SubprocVecEnv([make_env(i) for i in range(num_env)])


if __name__ == '__main__':
    # env = gym.make('grid_world-v0')
    env = make_vec_env('grid_world-v0', 8, 0)

    print("Www")


