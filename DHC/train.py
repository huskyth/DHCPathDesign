import argparse
import os
import random
import time

import torch
import numpy as np
import ray

from worker import GlobalBuffer, Learner, Actor
import configs

os.environ["OMP_NUM_THREADS"] = "1"
torch.manual_seed(2022)
np.random.seed(2022)
random.seed(2022)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./models/2000.pth', help='initial weights path')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    return parser.parse_args()


def main(opt=None, num_actors=configs.num_actors, log_interval=configs.log_interval):
    ray.init()
    # ray.init(local_mode=True)
    buffer = GlobalBuffer.remote()
    learner = Learner.remote(opt, buffer=buffer)
    time.sleep(1)
    actors = [Actor.remote(opt, i, 0.4 ** (1 + (i / (num_actors - 1)) * 7), learner, buffer) for i in range(num_actors)]

    for actor in actors:
        actor.run.remote()

    while not ray.get(buffer.ready.remote()):
        ray.get(learner.stats.remote(5))
        ray.get(buffer.stats.remote(5))

    print('start training')
    buffer.run.remote()
    learner.run.remote()

    done = False
    while not done:
        time.sleep(log_interval)
        done = ray.get(learner.stats.remote(log_interval))
        ray.get(buffer.stats.remote(log_interval))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    print("end")
