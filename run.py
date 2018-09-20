import gym

from baselines.a2c.runner import Runner
from baselines.a2c.a2c import Model

from baselines.common.policies import build_policy
import baselines.common.vec_env.subproc_vec_env as subproc_vec_env

import grid_world


"""
! !  ! ! !! ! ! ! ! ! !  ! ! ! !! 
! !  ! ! TODO: Proper Seeding !!! 
!! ! ! ! ! ! !  ! ! ! !!  ! ! ! ! 

First we will start with baselines to remove the possibility of 
getting the implementation wrong.
"""


def make_vec_env(env_id, num_env, seed):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            return env
        return _thunk

    return subproc_vec_env.SubprocVecEnv([make_env(i) for i in range(num_env)])


if __name__ == '__main__':
    env = make_vec_env('grid_world-v0', 8, 0)

    nsteps = 20

    # set_global_seeds(seed)

    nenvs = env.num_envs
    policy = build_policy(env, 'mike_cnn')

    model = Model(
        policy=policy,
        env=env,
        nsteps=nsteps,
        lrschedule='constant'
    )

    runner = Runner(env, model, nsteps=nsteps)

    while True:
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)

        print("www")

