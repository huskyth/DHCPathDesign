import numpy as np
from matplotlib import pyplot as plt
import torch

from DHC.configs import obs_radius

BLANK_AREA = [255, 255, 255, 255]
ENTRY_AREA = [0, 0, 0, 255]

BACKGROUND_INDEX = 2
BACKGROUND_AREA = [255, 255, 255, 0]

FOREGROUND_INDEX = 3
FOREGROUND_AREA = [255, 165, 0, 255]

AGENT_INDEX = 4
AGENT_AREA = [0, 191, 255, 255]

GOAL_INDEX = 5
GOAL_AREA = [255, 0, 0, 255]
color_map = np.array([BLANK_AREA,
                      ENTRY_AREA,
                      BACKGROUND_AREA,
                      FOREGROUND_AREA,
                      AGENT_AREA, GOAL_AREA
                      ],
                     dtype=np.uint8)


def show_two_map(first_frame, second_frame, origin_position, origin_goal,
                 is_show=True, is_save=True, name="shit.png"):
    # TODO:// position must be left top xy
    second_frame = np.array((second_frame * FOREGROUND_INDEX).detach().cpu().numpy(), dtype=int)
    first_frame, second_frame = color_map[first_frame], color_map[second_frame]
    x_length, y_length, dim = second_frame.shape
    start_x, start_y = origin_position[0], origin_position[1]
    goal_x, goal_y = origin_goal[0], origin_goal[1]

    zero_top = color_map[np.ones(first_frame.shape[:2], dtype=np.uint8) * BACKGROUND_INDEX]
    zero_top[start_x:start_x + x_length, start_y:start_y + y_length] = second_frame

    zero_top[start_x + obs_radius, start_y + obs_radius] = color_map[AGENT_INDEX]
    zero_top[goal_x + obs_radius, goal_y + obs_radius] = color_map[GOAL_INDEX]

    fig, ax = plt.subplots()
    ax.imshow(first_frame, animated=True, alpha=0.5)
    ax.imshow(zero_top, animated=True)
    plt.savefig(name)
    plt.show()
    plt.pause(0.5)


if __name__ == '__main__':
    np.random.seed(100)
    low_numpy = np.random.randint(low=0, high=2, size=(64, 30), dtype=np.uint8)
    high_frame = torch.tensor(np.random.randint(low=0, high=2, size=(9, 9), dtype=np.uint8))
    show_two_map(low_numpy, high_frame, np.array([5, 6]), np.array([10, 1]))
