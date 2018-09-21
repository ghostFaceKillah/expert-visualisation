import gym
import numpy as np

from baselines.a2c.runner import Runner
from baselines.a2c.a2c import Model
from baselines.common import set_global_seeds

from baselines.common.policies import build_policy
import baselines.common.vec_env.subproc_vec_env as subproc_vec_env
import baselines.common.vec_env.dummy_vec_env as dummy_ven_env

import grid_world
from grid_world.grid_world_env import generate_all_possible_state_obs
from grid_world.grid_constants import *
import grid_world.plotting as plotting


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

    # return subproc_vec_env.SubprocVecEnv([make_env(i) for i in range(num_env)])
    return dummy_ven_env.DummyVecEnv([make_env(i) for i in range(num_env)])


def check_value_function(model, s_o_pairs):
    obs_acc = [obs for state, obs in all_s_o_pairs]
    obs_batch = np.stack(obs_acc)

    probs, vals = model.evaluation(obs_batch)

    val_func = np.zeros_like(obs_batch[0, ..., 0], dtype=np.float32)
    for idx, (state, _) in enumerate(s_o_pairs):
        val_func[state.player_y, state.player_x] = vals[idx]

    for i in range(level.shape[0]):
        for j in range(level.shape[1]):
            if level[i, j] == PERIL:
                val_func[i, j] = -1
            if level[i, j] == DEADLY_PERIL:
                val_func[i, j] = -1
            if level[i, j] == GOAL:
                val_func[i, j] = 1

    return val_func


if __name__ == '__main__':
    env = make_vec_env('grid_world-v0', 8, 0)
    seed = 17
    nsteps = 20

    set_global_seeds(seed)

    nenvs = env.num_envs
    policy = build_policy(env, 'mike_cnn')

    level = env.envs[0].level
    all_s_o_pairs = generate_all_possible_state_obs(level)

    model = Model(
        policy=policy,
        env=env,
        nsteps=nsteps,
        lrschedule='constant',
        eval_size=len(all_s_o_pairs)
    )

    runner = Runner(env, model, nsteps=nsteps)
    r_acc = []

    for i in range(10000):
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)

        vf = check_value_function(model, all_s_o_pairs)
        if i % 20 == 0:
            plotting.PlotFuncOverLevel(
                level=level,
                player_y=None,
                player_x=None,
                func=vf
            ).save_img(f"imgs/update_{i:d}")


        r_mean = rewards.mean()
        r_acc.append(r_mean)

        print(r_mean)

