import os
import threading
import time
from copy import deepcopy

import torch
import ray
from torch.optim.lr_scheduler import MultiStepLR

from DHC import configs
from DHC.global_buffer import GlobalBuffer
from DHC.model import Network
from torch.optim import Adam

from DHC.utils.model_save_load_tool import model_save
from torch.cuda.amp import GradScaler
import torch.nn as nn


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, buffer: GlobalBuffer):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = Network()
        self.state = None
        self.weight_file = None

        self.model_save_counter = 0
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.avg_reward = 0
        self.avg_finish_cases = 0
        self.avg_step = 0
        self.model.to(self.device)
        self.tar_model = deepcopy(self.model)

        self.scheduler = MultiStepLR(self.optimizer, milestones=[200000, 400000], gamma=0.5)
        self.buffer = buffer

        self.done = False

        self.store_weights()

        self.learning_thread = None
        self.weights_id = None

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        self.weights_id = ray.put(state_dict)

    def run(self):
        self.learning_thread = threading.Thread(target=self.train, daemon=True)
        self.learning_thread.start()

    def train(self):
        scaler = GradScaler()

        while not ray.get(self.buffer.check_done.remote()):

            for i in range(1, 100001):
                data_id = ray.get(self.buffer.get_data.remote())
                data = ray.get(data_id)
                b_obs, b_action, b_reward, b_done, b_steps, b_seq_len, b_hidden, b_comm_mask, \
                    idxes, weights, old_ptr = data

                b_obs, b_action, b_reward = b_obs.to(self.device), b_action.to(self.device), \
                    b_reward.to(self.device)
                b_done, b_steps, weights = b_done.to(self.device), b_steps.to(self.device), \
                    weights.to(self.device)
                b_hidden = b_hidden.to(self.device)
                b_comm_mask = b_comm_mask.to(self.device)

                b_next_seq_len = [(seq_len + forward_steps).item() for seq_len, forward_steps in
                                  zip(b_seq_len, b_steps)]
                b_next_seq_len = torch.LongTensor(b_next_seq_len)

                with torch.no_grad():
                    b_q_ = (1 - b_done) * self.tar_model(b_obs, b_next_seq_len, b_hidden,
                                                         b_comm_mask).max(1, keepdim=True)[0]

                b_q = self.model(b_obs[:, :-configs.forward_steps], b_seq_len, b_hidden,
                                 b_comm_mask[:, :-configs.forward_steps]).gather(1, b_action)

                td_error = (b_q - (b_reward + (0.99 ** b_steps) * b_q_))

                priorities = td_error.detach().squeeze().abs().clamp(1e-4).cpu().numpy()

                loss = (weights * self.huber_loss(td_error)).mean()

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()

                scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 40)

                scaler.step(self.optimizer)
                scaler.update()

                self.scheduler.step()

                if i % 5 == 0:
                    self.store_weights()

                self.buffer.update_priorities.remote(idxes, priorities, old_ptr)

                self.model_save_counter += 1

                # update target net, save model
                if i % configs.target_network_update_freq == 0:
                    self.tar_model.load_state_dict(self.model.state_dict())

                if i % configs.save_interval == 0:
                    now_time = time.strftime("%Y-%m-%d-%H", time.localtime())
                    path = os.path.join(configs.save_path, '{}-{}.pth'.format(now_time,
                                                                              self.model_save_counter))
                    model_save(self.model, path)
                    print(
                        "save model path:" + os.path.join(configs.save_path, '{}-{}.pth'.
                                                          format(now_time, self.model_save_counter)))

        self.done = True

    def huber_loss(self, td_error, kappa=1.0):
        abs_td_error = td_error.abs()
        flag = (abs_td_error < kappa).float()
        return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)

    def stats(self, interval: int):
        return self.done
