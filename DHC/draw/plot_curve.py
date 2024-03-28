from time import strftime, localtime, time
import numpy as np
from matplotlib import pyplot as plt


def plot_curve(self, result, title):
    plt.figure()
    title = title
    plt.title(title)
    plt.grid()
    plt.plot(result, marker='*', linestyle="-")
    plt.xlabel('times')
    plt.ylabel('reward')
    currentTime = strftime("%Y-%m-%d-%H", localtime(time()))
    filename = "avgcurve&" + "@" + currentTime
    filepath = self.output['-dir'] + 'IncreasedCnt/'
    plt.savefig(filepath + filename)
    plt.show()


def plot_mul_curve(result):
    plt.figure()
    plt.grid()
    dict_len = len(result[1])
    x = list(range(dict_len))
    plt.xticks(np.arange(0, dict_len, 1))
    label_name = {0: 'ade_avg_12', 1: 'fde_avg_12', 2: 'ade_min_12', 3: 'fde_min_12'}
    for i in range(len(result)):
        y = result[i]
        plt.plot(x, y, marker='*', linestyle="-", label=label_name[i])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    title = "epoch_loss"
    plt.title(title)
    plt.legend()
    file = './result/loss_curve&{}'.format(strftime('%Y-%m-%d %H:%M:%S', localtime(time())))
    plt.savefig(file)
    plt.show()


if __name__ == '__main__':
    res = [[1, 2, 3], [4, 5, 5], [4, 7, 8], [4, 5, 6]]
    plot_mul_curve(res)
