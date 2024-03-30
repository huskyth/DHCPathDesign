import random

import numpy as np
import ray
import torch

from DHC import configs
from DHC.buffer import LocalBuffer
from DHC.global_buffer import GlobalBuffer
from DHC.learner import Learner
from DHC.model import Network
from dyn_environment import Environment


@ray.remote(num_cpus=1)
class Actor:
    def __init__(self, worker_id: int, epsilon: float, learner: Learner, buffer: GlobalBuffer, summary):
        self.id = worker_id
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = Network()

        self.model.eval()
        self.env = Environment(curriculum=False)
        self.epsilon = epsilon
        self.learner = learner
        self.global_buffer = buffer
        self.max_episode_length = configs.max_episode_length
        self.update_counter = 0

        self.my_summary = summary
        self.epoch = 0

    def run(self):
        obs, pos, local_buffer = self.reset()
        while True:
            actions, q_val, hidden, comm_mask = self.model.step(torch.from_numpy(obs.astype(np.float32)),
                                                                torch.from_numpy(pos.astype(np.float32)))
            if random.random() < self.epsilon:
                # Note: only one agent do random action in order to keep the environment stable
                actions[0] = np.random.randint(0, 5)
            (next_obs, next_pos), rewards, done, _ = self.env.step(actions)
            local_buffer.add(q_val[0], actions[0], rewards[0], next_obs, hidden, comm_mask)
            if done is False and self.env.steps < self.max_episode_length:
                obs, pos = next_obs, next_pos
            else:
                if done:
                    data = local_buffer.finish()
                    print("done~~~")

                else:
                    _, q_val, hidden, comm_mask = self.model.step(torch.from_numpy(next_obs.astype(np.float32)),
                                                                  torch.from_numpy(next_pos.astype(np.float32)))
                    data = local_buffer.finish(q_val[0], comm_mask)
                return_value = data[-1]
                self.my_summary.add_float.remote(x=self.epoch + 1, y=return_value, title="Return Value",
                                          x_name=f"{self.id}_epoch")
                self.global_buffer.add.remote(data)
                obs, pos, local_buffer = self.reset()
                self.epoch += 1

            self.update_counter += 1

            if self.update_counter == configs.actor_update_steps:
                self.update_weights()
                self.update_counter = 0

    def update_weights(self):
        """load weights from learner"""
        weights_id = ray.get(self.learner.get_weights.remote())
        weights = ray.get(weights_id)
        self.model.load_state_dict(weights)
        new_env_settings_set = ray.get(self.global_buffer.get_env_settings.remote())
        self.env.update_env_settings_set(ray.get(new_env_settings_set))

    def reset(self):
        self.model.reset()
        obs, pos = self.env.reset()
        local_buffer = LocalBuffer(self.id, self.env.num_agents, self.env.map_size[0], obs)
        return obs, pos, local_buffer
