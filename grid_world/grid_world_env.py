import attr
import gym
from gym.utils import seeding
import numpy as np
import os

import constants

# map encoding meanings
from grid_world.plotting import _state_to_human_array
from grid_world.grid_constants import *


def load(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()

    # Strip whitespace
    new_lines = []
    for line in lines:
        new_lines.append(line.strip())
    lines = new_lines

    # Make sure they are all the same length
    first_len = len(lines[0])

    for line in lines:
        assert len(line) == first_len

    shape = (
        len(lines),
        first_len
    )

    level = np.zeros(shape=shape, dtype=np.uint8)

    for i, line in enumerate(lines):
        for j, item in enumerate(line):
            level[i][j] = int(item)

    return level


@attr.s(auto_attribs=True)
class State:
    player_x: int
    player_y: int

    level: np.ndarray


def _state_to_obs(level, state):
    """
    Map state of environment to an observation that can be
    consumed by a neural net.
    """
    walls = (level == WALL).astype(np.uint8)
    player_pos = np.zeros_like(level)
    player_pos[state.player_y, state.player_x] = 1
    perils = (level == PERIL).astype(np.uint8)
    goals = (level == GOAL).astype(np.uint8)

    resu = np.stack((walls, player_pos, perils, goals))
    resu = np.moveaxis(resu, 0, -1)

    return resu


def generate_all_possible_state_obs(level):
    acc = []
    for i in range(level.shape[0]):
        for j in range(level.shape[1]):
            if level[i, j] not in PRINTALBLE:
                state = State(
                    player_x=j,
                    player_y=i,
                    level=level
                )
                acc.append((state, _state_to_obs(level, state)))

    return acc


def _generate_fname():
    return os.path.join(constants.ROOT_DIR, 'levels', 'level_03.txt')


class GridWorldEnv(gym.Env):
    def __init__(self):
        fname = _generate_fname()
        self.level = load(fname)

        self.obs_shape = self.level.shape + (4,)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(4)

        self.state = self._new_state()
        self.viewer = None


    def _draw_player_initial_position(self):
        acc = []
        for i in range(self.level.shape[0]):
            for j in range(self.level.shape[1]):
                if self.level[i][j] == START:
                    acc.append((i, j))
        player_y, player_x = acc[np.random.randint(len(acc))]

        return player_x, player_y

    def _new_state(self):
        player_x, player_y = self._draw_player_initial_position()

        return State(
            player_y=player_y,
            player_x=player_x,
            level=self.level
        )

    def _map_action_to_new_position(self, player_x, player_y, action):
        if action == LEFT:
            new_x, new_y = player_x - 1, player_y
        elif action == RIGHT:
            new_x, new_y = player_x + 1, player_y
        elif action == UP:
            new_x, new_y = player_x, player_y - 1
        elif action == DOWN:
            new_x, new_y = player_x, player_y + 1
        else:
            raise ValueError("Unrecognized action")

        new_x = max(new_x, 0)
        new_y = max(new_y, 0)

        new_x = min(new_x, self.level.shape[1] - 1)
        new_y = min(new_y, self.level.shape[0] - 1)

        return new_x, new_y

    def _handle(self, field_type: int):
        if field_type == FREE_PASS:
            resu = (0, False, True)
        elif field_type == START:
            resu = (0, False, True)
        elif field_type == WALL:
            resu = (0, False, False)
        elif field_type == GOAL:
            resu = (1, True, True)
        elif field_type == PERIL:
            resu = (-1, False, True)
        elif field_type == DEADLY_PERIL:
            resu = (-1, True, False)
        else:
            raise ValueError(f"Unknown field type {field_type}")

        reward, done, can_move_here = resu

        return reward, done, can_move_here

    def step(self, action):
        # map action to new position
        new_x, new_y = self._map_action_to_new_position(
            self.state.player_x,
            self.state.player_y,
            action
        )

        # handle field we came across
        field_type = self.level[new_y, new_x]
        reward, done, can_move_here = self._handle(field_type)

        if can_move_here:
            self.state.player_x = new_x
            self.state.player_y = new_y

        obs = _state_to_obs(self.level, self.state)

        return obs, reward, done, {}

    def reset(self):
        self.state = self._new_state()
        return _state_to_obs(self.level, self.state)

    def render(self, mode='rgb'):
        if mode == 'rgb':
            return _state_to_human_array(
                self.level, self.state.player_x, self.state.player_y
            )
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            img = _state_to_human_array(
                self.level, self.state.player_x, self.state.player_y
            )
            self.viewer.imshow(img)
            return self.viewer.isopen
        else:
            raise ValueError("Unknown display mode")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def _map_input_to_action(key):
    if key == 'w':
        return UP
    elif key == 's':
        return DOWN
    elif key == 'a':
        return LEFT
    elif key == 'd':
        return RIGHT


if __name__ == '__main__':
    env = GridWorldEnv()
    env.reset()
    env.render(mode='human')
    done = False

    while not done:
        key = input("What's next men")
        action = _map_input_to_action(key)

        print(f"Taking action {action}")

        obs, reward, done, info = env.step(action)
        if reward != 0:
            print(f"Got reward {reward}!")
        env.render(mode='human')

    env.close()

