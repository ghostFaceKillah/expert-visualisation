import numpy as np

from grid_world.grid_world_env import load, _generate_fname
from grid_world.plotting import PlotFuncOverLevel


if __name__ == '__main__':
    level = load(_generate_fname())

    f_plot = PlotFuncOverLevel(
        level,
        player_x=0,
        player_y=0,
        func=np.random.uniform(high=1.0, low=0.0, size=level.shape)
    )
    f_plot.plot()

    print("ww")