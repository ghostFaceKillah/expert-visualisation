import attr
import gym
import numpy as np
import os

import constants

# map encoding meanings
FREE_PASS = 0
WALL = 1
PLAYER = 2
GOAL = 3
PERIL = 4
DEADLY_PERIL = 5

COLOR_MAP = {
    WALL: [100, 100, 100],
    PERIL: [255, 255, 0],
    DEADLY_PERIL: [255, 0, 0],
    GOAL:  [0, 255, 0],
    PLAYER: [0, 0, 255]
}

# actions
NONE = 0
LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4


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

PIXEL_SIZE = 10


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


class GridWorldEnv(gym.Env):
    def __init__(self):
        fname = self._generate_fname()
        self.level = load(fname)

        self.obs_shape = self.level.shape + (4,)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)

        self.state = self._new_state()
        self.viewer = None

    def _generate_fname(self):
        return os.path.join(constants.ROOT_DIR, 'levels', 'level_01.txt')

    def _draw_player_initial_position(self):
        while True:
            player_y = np.random.randint(self.level.shape[0])
            player_x = np.random.randint(self.level.shape[1])

            if self.level[player_y][player_x] == 0:
                return player_x, player_y

    def _new_state(self):
        player_x, player_y = self._draw_player_initial_position()

        return State(
            player_y=player_y,
            player_x=player_x,
            level=self.level
        )

    def _map_action_to_new_position(self, player_x, player_y, action):
        if action == NONE:
            new_x, new_y = player_x, player_y
        elif action == LEFT:
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

    def _state_to_human_array(self):
        img_shape = self.level.shape + (3,)
        img = np.zeros(shape=img_shape, dtype=np.uint8)

        for thing, color in COLOR_MAP.items():
            img[self.level == thing] = color

        img[self.state.player_y, self.state.player_x] = COLOR_MAP[PLAYER]

        # Make it not for ants
        big_img = np.repeat(img, PIXEL_SIZE, axis=0)
        big_img = np.repeat(big_img, PIXEL_SIZE, axis=1)

        return big_img

    def step(self, action):
        """

        """
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

    def render(self, mode='human'):
        # from gym.envs.classic_control import rendering
        # if self.viewer is None:
        #     self.viewer = rendering.SimpleImageViewer()
        # img = self._state_to_human_array()
        # self.viewer.imshow(img)
        # return self.viewer.isopen
        return self._state_to_human_array()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        pass


def _map_input_to_action(key):
    if key == 'w':
        return UP
    elif key == 's':
        return DOWN
    elif key == 'a':
        return LEFT
    elif key == 'd':
        return RIGHT
    else:
        return NONE


if __name__ == '__main__':
    env = GridWorldEnv()
    env.reset()
    env.render()
    done = False

    while not done:
        key = input("What's next men")
        action = _map_input_to_action(key)

        print(f"Taking action {action}")

        obs, reward, done, info = env.step(action)
        if reward != 0:
            print(f"Got reward {reward}!")
        env.render()

    env.close()

