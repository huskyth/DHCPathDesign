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


def main(num_actors=configs.num_actors, log_interval=configs.log_interval):
    ray.init()
    # ray.init(local_mode=True)
    buffer = GlobalBuffer.remote()
    my_summary = MySummary(use_wandb=False)
    learner = Learner.remote(buffer=buffer, summary=my_summary)
    time.sleep(1)
    actors = [Actor.remote(i, 0.4 ** (1 + (i / (num_actors - 1)) * 7),
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


if __name__ == '__main__':
    main()
