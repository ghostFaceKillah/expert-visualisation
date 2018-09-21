import numpy as np
import matplotlib as mpl
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

def relative_luminance(color):
    """Calculate the relative luminance of a color according to W3C standards
    Taken straight from seaborn.
    Parameters
    ----------
    color : matplotlib color or sequence of matplotlib colors
        Hex code, rgb-tuple, or html color name.
    Returns
    -------
    luminance : float(s) between 0 and 1
    """
    rgb = mpl.colors.colorConverter.to_rgba_array(color)[:, :3]
    rgb = np.where(rgb <= .03928, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    lum = rgb.dot([.2126, .7152, .0722])
    try:
        return lum.item()
    except ValueError:
        return lum


def _to_center(sides):
    x = (sides[0][0] + sides[2][0]) / 2
    y = (sides[0][1] + sides[2][1]) / 2

    return x, y


class Square:
    def __init__(self, level, idx, idy, value, normalized_value, cmap, fmt=None):

        self.level = level
        self.idx = idx
        self.idy = idy
        self.value = value  # should be already normalized
        self.normalized_value = normalized_value
        self.cmap = cmap

        self.sides = _map_to_unit_square(
            self.level.shape[1],
            self.level.shape[0],
            idx,
            idy
        )

        self.center = _to_center(self.sides)

        self.color = self._resolve_color()
        if fmt is None:
            self.fmt = ".2f"
        else:
            self.fmt = fmt

    def _resolve_color(self):
        type = self.level[self.idy, self.idx]

        if type in PRINTALBLE:
            raw_color = COLOR_MAP[type]
            color = [float(c) / 255 for c in raw_color]
            return color
        else:
            return self.cmap(self.normalized_value)

    def _annotate_text(self, ax):
        lum = relative_luminance(self.color)
        text_color = ".15" if lum > .408 else "w"
        annotation = ("{:" + self.fmt + "}").format(self.value)
        text_kwargs = dict(color=text_color, ha="center", va="center")
        x, y = self.center
        ax.text(x, y, annotation, **text_kwargs)

    def plot(self, ax):
        polygon = plt.Polygon(self.sides, facecolor=self.color)
        ax.add_patch(polygon)
        self._annotate_text(ax)

    def __repr__(self):
        return str(self.sides)


class PlotFuncOverLevel:
    def __init__(self, level: np.ndarray, player_x, player_y, func: np.ndarray, scaling='minus_unit'):
        assert level.shape == func.shape

        self.level = level

        self.player_x = player_x
        self.player_y = player_y

        self.func =func
        if scaling == 'minus_unit':
            self.norm_func = (func + 1) / 2

        self.cmap = plt.get_cmap('coolwarm')

        fields = []

        for y in range(level.shape[0]):
            for x in range(level.shape[1]):
                value = func[y, x]
                norm_val = self.norm_func[y, x]
                sq = Square(level, x, y, value, norm_val, self.cmap)
                fields.append(sq)

        self.fields = fields

    def save_img(self, img_fname):
        ax = plt.axes()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for field in self.fields:
            field.plot(ax)
        plt.savefig(img_fname)
        plt.close()

