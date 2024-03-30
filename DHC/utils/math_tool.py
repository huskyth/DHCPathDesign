import random

import math as mh

import numpy as np
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
file = str(FILE.parent)
if file not in sys.path:
    sys.path.append(file)

from coordinate_tool import *

x_translation = lambda x: mh.log((1 - x) / x)
X_VARIATION = 5000


def rank_agent_by_distance(agent_position, target_position):
    assert len(agent_position) == len(target_position)
    d = {}
    for i in range(len(agent_position)):
        d[i + 1] = cal_distance_between_2_point(agent_position[i][0], agent_position[i][1], target_position[i][0],
                                                target_position[i][1])

    res = sorted(d.items(), key=lambda x: x[1])
    return [x[0] for x in res]


if __name__ == '__main__':
    x_test = np.arange(0, 50000, 1)
    agents_pos = np.empty((4, 2), dtype=int)
    goals_pos = np.empty((4, 2), dtype=int)
    for x in range(4):
        agents_pos[x][0], agents_pos[x][1] = random.randint(0, 3), random.randint(0, 3)
        goals_pos[x][0], goals_pos[x][1] = random.randint(0, 3), random.randint(0, 3)
    print(agents_pos)
    print(goals_pos)
    r = rank_agent_by_distance(agents_pos, goals_pos)
    print(r)
