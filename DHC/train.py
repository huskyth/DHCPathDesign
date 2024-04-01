import os
import random
import time

import torch
import numpy as np
import ray

from DHC.actor import Actor
from DHC.learner import Learner
from DHC.utils.tensor_board_tool import MySummary
from global_buffer import GlobalBuffer
import configs

os.environ["OMP_NUM_THREADS"] = "1"
torch.manual_seed(2022)
np.random.seed(2022)
random.seed(2022)

DEBUG_MODE = False


def ray_init():
    if not DEBUG_MODE:
        ray.init(_temp_dir="/home/husky/Desktop/tmp")
    else:
        ray.init(local_mode=True)


def epsilon():
    return -1 if not DEBUG_MODE else -1 + 1e-5


def main(num_actors=configs.num_actors, log_interval=configs.log_interval):
    ray_init()
    buffer = GlobalBuffer.remote()
    my_summary = MySummary.remote(use_wandb=False)
    learner = Learner.remote(buffer=buffer, summary=my_summary)
    time.sleep(1)
    actors = [Actor.remote(i, 0.4 ** (1 + (i / (num_actors + epsilon())) * 7),
                           learner, buffer, my_summary) for i in range(num_actors)]

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

# https://docs.ray.io/en/latest/ray-core/scheduling/ray-oom-prevention.html OOM
if __name__ == '__main__':
    main()
