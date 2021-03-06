
FREE_PASS = 0
WALL = 1
START = 2
GOAL = 3
PERIL = 4
DEADLY_PERIL = 5
PLAYER = 6

# actions
# NONE = 0
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3

PIXEL_SIZE = 10

COLOR_MAP = {
    WALL: [100, 100, 100],
    START: [50, 50, 50],
    PERIL: [255, 255, 0],
    DEADLY_PERIL: [255, 0, 0],
    GOAL:  [0, 255, 0],
    PLAYER: [0, 0, 255]
}

PRINTALBLE = {
    WALL, PERIL, DEADLY_PERIL, GOAL
}

SIDES = [
    'left',
    'right',
    'up',
    'down'
]
