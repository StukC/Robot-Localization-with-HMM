import numpy as np
height = 6
width = 7
obstacles = ((1, 2),
             (1, 3),
             (1, 4),
             (1, 5),
             (2, 2),
             (2, 5),
             (3, 1),
             (3, 2),
             (3, 5),
             (4, 4),
             (4, 5))
# E = Empty, # = Obstacle
#   0 1 2 3 4 5 6
# 0 E E E E E E E
# 1 E E # # # # E
# 2 E E # E E # E
# 3 E # # E E # E
# 4 E E E E # # E
# 5 E E E E E E E
DETECT_WALL = 0.85
FAIL_TO_DETECT_WALL = 0.15
DETECT_OPEN = 0.9
FAIL_TO_DETECT_OPEN = 0.1

NORTH = 1
EAST = 2

# robot will perform following sequence
sequence = [[0, 0, 0, 1],
            NORTH,
            [0, 1, 0, 0],
            EAST,
            [0, 1, 0, 0],
            EAST,
            [0, 0, 1, 0]]

# places open spaces
open_spaces = [(x, y) for x in range(6) for y in range(7) if (x, y) not in obstacles]

sequence_item = 'evidence'
PROB_GO_FORWARD = 0.8
PROB_DRIFT = 0.1
