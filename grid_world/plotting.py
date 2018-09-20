import numpy as np
import matplotlib.pyplot as plt

from grid_world.grid_constants import *


def _state_to_human_array(level, player_x, player_y):
    img_shape = level.shape + (3,)
    img = np.zeros(shape=img_shape, dtype=np.uint8)

    for thing, color in COLOR_MAP.items():
        img[level == thing] = color

    img[player_y, player_x] = COLOR_MAP[PLAYER]

    # Make it not for ants
    big_img = np.repeat(img, PIXEL_SIZE, axis=0)
    big_img = np.repeat(big_img, PIXEL_SIZE, axis=1)

    return big_img


def _plot_func_over_level(level: np.ndarray, player_x: int, player_y: int, f: np.ndarray):
    """
    """
    pass


def _map_to_unit_square(len_x, len_y, idx, idy):
    longer = max(len_x, len_y)

    offset_x = (longer - len_x) / 2
    offset_y = (longer - len_y) / 2

    unit = 1.0 / longer

    x_start = (idx + offset_x) * unit
    y_start = (idy + offset_y) * unit

    return [
        [x_start, y_start],
        [x_start + unit, y_start],
        [x_start + unit, y_start + unit],
        [x_start, y_start + unit],
    ]


class Square:
    def __init__(self, level, idx, idy, value):

        self.level = level
        self.idx = idx
        self.idy = idy
        self.value = value

        self.sides = _map_to_unit_square(
            self.level.shape[1],
            self.level.shape[0],
            idx,
            idy
        )

        self.color = self._resolve_color()

    def _resolve_color(self):
        type = self.level[self.idy, self.idx]

        if type in PRINTALBLE:
            raw_color = COLOR_MAP[type]
            color = [float(c) / 255 for c in raw_color]
            return color
        else:
            return self.value * np.ones(shape=3, dtype=np.float32)

    def plot(self, ax):
        polygon = plt.Polygon(self.sides, facecolor=self.color)
        ax.add_patch(polygon)

    def __repr__(self):
        return str(self.sides)


class PlotFuncOverLevel:
    def __init__(self, level: np.ndarray, player_x: int, player_y: int, func: np.ndarray):
        assert level.shape == func.shape

        self.level = level

        self.player_x = player_x
        self.player_y = player_y

        self.func = func

        fields = []

        for y in range(level.shape[0]):
            for x in range(level.shape[1]):
                value = func[y, x]
                sq = Square(level, x, y, value)
                fields.append(sq)

        self.fields = fields

    def plot(self):
        ax = plt.axes()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for field in self.fields:
            field.plot(ax)
        plt.savefig('hehe.png')
        plt.close()


