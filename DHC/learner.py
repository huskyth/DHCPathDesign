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

from DHC.test_dyn import test_while_training
from DHC.utils.info import print_process_info
from DHC.utils.model_save_load_tool import RESUME_MODEL_NAME, model_load, model_save
from DHC.utils.visual import init_summary_writer, plot
from torch.cuda.amp import GradScaler
import torch.nn as nn


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, opt, buffer: GlobalBuffer):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = Network()
        self.state = None
        self.weight_file = None

        self.counter = 0
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)
        self.avg_loss = 0
        self.avg_reward = 0
        self.avg_finish_cases = 0
        self.avg_step = 0
        if RESUME_MODEL_NAME:
            self.weight_file = os.path.join(configs.save_path, RESUME_MODEL_NAME)
            self.state = model_load(self.weight_file, self.device)
            self.model.load_state_dict(self.state['model_state'])
            self.counter = self.state['counter']
            self.avg_loss = self.state['avg_loss']
            self.avg_reward = self.state['avg_reward']
            self.avg_finish_cases = self.state['avg_finish_cases']
            self.avg_step = self.state['avg_step']
            print('lode model from {}'.format(RESUME_MODEL_NAME), self.counter, self.avg_reward, self.avg_loss)

        self.model.to(self.device)
        self.tar_model = deepcopy(self.model)

        self.scheduler = MultiStepLR(self.optimizer, milestones=[200000, 400000], gamma=0.5)
        self.buffer = buffer

        self.last_counter = 0
        self.done = False
        self.loss = 0
        self.writer = init_summary_writer()

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
        accumulative_reward = self.avg_reward * self.counter
        accumulative_loss = self.avg_loss * self.counter
        accumulative_avg_finish_cases = self.avg_finish_cases * self.counter
        accumulative_avg_step = self.avg_step * self.counter
        scaler = GradScaler()

        while not ray.get(self.buffer.check_done.remote()) and self.counter < configs.training_times:

            for i in range(1, 100001):
                data_id = ray.get(self.buffer.get_data.remote())
                data = ray.get(data_id)
                b_obs, b_action, b_reward, b_done, b_steps, b_seq_len, b_hidden, b_comm_mask, idxes, weights, old_ptr = data

                print_process_info(0)

                mean_reward = b_reward.mean().item()
                accumulative_reward += mean_reward
                self.avg_reward = accumulative_reward / (self.counter + 1)
                plot(self.writer, x=self.counter + 1, y=self.avg_reward,
                     title='Average reward per counter')
                plot(self.writer, x=self.counter + 1, y=mean_reward,
                     title='Reward per counter')

                b_obs, b_action, b_reward = b_obs.to(self.device), b_action.to(self.device), b_reward.to(self.device)
                b_done, b_steps, weights = b_done.to(self.device), b_steps.to(self.device), weights.to(self.device)
                b_hidden = b_hidden.to(self.device)
                b_comm_mask = b_comm_mask.to(self.device)

                b_next_seq_len = [(seq_len + forward_steps).item() for seq_len, forward_steps in
                                  zip(b_seq_len, b_steps)]
                b_next_seq_len = torch.LongTensor(b_next_seq_len)

                with torch.no_grad():
                    b_q_ = (1 - b_done) * \
                           self.tar_model(b_obs, b_next_seq_len, b_hidden, b_comm_mask).max(1, keepdim=True)[0]

                b_q = self.model(b_obs[:, :-configs.forward_steps], b_seq_len, b_hidden,
                                 b_comm_mask[:, :-configs.forward_steps]).gather(1, b_action)

                td_error = (b_q - (b_reward + (0.99 ** b_steps) * b_q_))

                priorities = td_error.detach().squeeze().abs().clamp(1e-4).cpu().numpy()

                loss = (weights * self.huber_loss(td_error)).mean()
                self.loss += loss.item()

                accumulative_loss += loss.item()
                self.avg_loss = accumulative_loss / (self.counter + 1)
                plot(self.writer, x=self.counter + 1, y=self.avg_loss,
                     title='Average loss per counter')

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()

                scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 40)

                scaler.step(self.optimizer)
                scaler.update()

                self.scheduler.step()

                # store new weights in shared memory
                if i % 5 == 0:
                    self.store_weights()

                self.buffer.update_priorities.remote(idxes, priorities, old_ptr)

                self.counter += 1

                avg_is_finish, avg_steps = test_while_training(self.model, num=1)
                accumulative_avg_finish_cases += avg_is_finish
                self.avg_finish_cases = accumulative_avg_finish_cases / self.counter
                accumulative_avg_step += avg_steps
                self.avg_step = accumulative_avg_step / self.counter
                plot(self.writer, x=self.counter, y=self.avg_finish_cases,
                     title='Average accumulate finishing cases per batch')
                plot(self.writer, x=self.counter, y=self.avg_step,
                     title='Average accumulate steps per batch')

                # update target net, save model
                if i % configs.target_network_update_freq == 0:
                    self.tar_model.load_state_dict(self.model.state_dict())

                if i % configs.save_interval == 0:
                    nowtime = time.strftime("%Y-%m-%d-%H", time.localtime())
                    path = os.path.join(configs.save_path, '{}-{}.pth'.format(nowtime, self.counter))
                    model_save(self.model, path, self.counter, self.avg_loss, self.avg_reward, self.avg_finish_cases,
                               self.avg_steps)
                    print(
                        "save model path:" + os.path.join(configs.save_path, '{}-{}.pth'.format(nowtime, self.counter)))

        self.done = True
        self.writer.close()

    def huber_loss(self, td_error, kappa=1.0):
        abs_td_error = td_error.abs()
        flag = (abs_td_error < kappa).float()
        return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)

    def log(self, interval, use=False):
        if not use: return
        print('number of updates: {}'.format(self.counter))
        print('update speed: {}/s'.format((self.counter - self.last_counter) / interval))
        if self.counter != self.last_counter:
            print('loss: {:.4f}'.format(self.loss / (self.counter - self.last_counter)))

    def stats(self, interval: int):
        self.log(interval)
        self.last_counter = self.counter
        self.loss = 0
        return self.done
