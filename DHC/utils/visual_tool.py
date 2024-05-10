import numpy as np
from matplotlib import pyplot as plt

color_map = np.array([[255, 255, 255, 255],  # white
                      [190, 190, 190, 255],  # gray
                      [0, 191, 255, 255],  # blue
                      [255, 165, 0, 255],  # orange
                      [0, 250, 154, 255],  # green
                      [0, 0, 0, 255]],
                     dtype=np.uint8)


def show_two_map(first_frame, second_frame, position):
    # TODO:// position must be left top xy
    second_frame = np.array(second_frame.numpy(), dtype=np.int)
    first_frame, second_frame = color_map[first_frame], color_map[second_frame]
    x_length, y_length, dim = second_frame.shape
    start_x, start_y = position[0], position[1]
    zero_top = np.zeros(first_frame.shape, dtype=np.uint8)
    zero_top[start_x:start_x + x_length, start_y:start_y + y_length] = second_frame

    fig, ax = plt.subplots()
    ax.imshow(first_frame, animated=True)
    ax.imshow(zero_top, animated=True, alpha=0.5)
    plt.show()
    plt.pause(0.5)


if __name__ == '__main__':
    np.random.seed(100)
    low_numpy = np.random.randint(low=0, high=2, size=(54, 30), dtype=np.uint8)
    high_frame = np.random.randint(low=3, high=4, size=(9, 9), dtype=np.uint8)
    show_two_map(low_numpy, high_frame, np.array([5, 6]))
