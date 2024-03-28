import os
import random

import numpy as np
import ray
import torch

from DHC import configs
from DHC.buffer import LocalBuffer
from DHC.global_buffer import GlobalBuffer
from DHC.learner import Learner
from DHC.model import Network
from DHC.utils.math_tool import epsilon_decay
from DHC.utils.model_save_load_tool import RESUME_MODEL_NAME, model_load
from DHC.utils.visual import init_summary_writer, plot
from dyn_environment import Environment


@ray.remote(num_cpus=1)
class Actor:
    def __init__(self, opt, worker_id: int, epsilon: float, learner: Learner, buffer: GlobalBuffer):
        self.id = worker_id
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = Network()

        if RESUME_MODEL_NAME:
            self.weight_file = os.path.join(configs.save_path, RESUME_MODEL_NAME)
            self.state = model_load(self.weight_file, self.device)
            print('lode model from {}'.format(RESUME_MODEL_NAME))
            self.model.load_state_dict(self.state['model_state'])

        self.model.eval()
        self.env = Environment(curriculum=False)
        self.epsilon = epsilon
        self.learner = learner
        self.global_buffer = buffer
        self.max_episode_length = configs.max_episode_length
        self.counter = 0
        self.writer = init_summary_writer()

    def run(self):
        obs, pos, local_buffer = self.reset()
        epsilon_count = 0
        accumulate_reward_per_episode = 0
        while True:
            epsilon_use = epsilon_decay(self.epsilon, epsilon_count)
            # sample action
            actions, q_val, hidden, comm_mask = self.model.step(torch.from_numpy(obs.astype(np.float32)),
                                                                torch.from_numpy(pos.astype(np.float32)))
            if random.random() < epsilon_use:
                # Note: only one agent do random action in order to keep the environment stable
                actions[0] = np.random.randint(0, 5)
            # take action in env
            (next_obs, next_pos), rewards, done, _ = self.env.step(actions)  # 这里的reward是用上了的
            local_buffer.add(q_val[0], actions[0], rewards[0], next_obs, hidden, comm_mask)
            accumulate_reward_per_episode += rewards[0]
            if done is False and self.env.steps < self.max_episode_length:
                obs, pos = next_obs, next_pos
            else:
                # finish and send buffer
                if done:
                    data = local_buffer.finish()
                    print("done~~~")

                else:
                    _, q_val, hidden, comm_mask = self.model.step(torch.from_numpy(next_obs.astype(np.float32)),
                                                                  torch.from_numpy(next_pos.astype(np.float32)))
                    data = local_buffer.finish(q_val[0], comm_mask)
                plot(self.writer, epsilon_count + 1, accumulate_reward_per_episode, 'Reward per episode')
                epsilon_count += 1
                accumulate_reward_per_episode = 0
                self.global_buffer.add.remote(data)
                obs, pos, local_buffer = self.reset()

            self.counter += 1

            if self.counter == configs.actor_update_steps:
                self.update_weights()
                self.counter = 0

    def update_weights(self):
        """load weights from learner"""
        # update network parameters
        weights_id = ray.get(self.learner.get_weights.remote())
        weights = ray.get(weights_id)
        self.model.load_state_dict(weights)
        # update environment settings set (number of agents and map size)
        new_env_settings_set = ray.get(self.global_buffer.get_env_settings.remote())
        self.env.update_env_settings_set(ray.get(new_env_settings_set))

    def reset(self):
        self.model.reset()
        obs, pos = self.env.reset()
        local_buffer = LocalBuffer(self.id, self.env.num_agents, self.env.map_size[0], obs)
        return obs, pos, local_buffer
