import os
import threading
import time
from copy import deepcopy

import torch
import ray
from torch.optim.lr_scheduler import MultiStepLR

from DHC import configs
from DHC.global_buffer import GlobalBuffer
from DHC.icm.icm_model import ICM
from DHC.model import Network
from torch.optim import Adam

from DHC.utils.model_save_load_tool import model_save
from torch.cuda.amp import GradScaler
import torch.nn as nn

import torch.nn.functional as F

from DHC.utils.visual_tool import show_two_map


@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, buffer: GlobalBuffer, summary, map):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = Network()
        self.icm = ICM().to(device=self.device)
        self.map = map
        self.state = None
        self.weight_file = None

        self.optimizer = Adam(list(self.model.parameters()) + list(self.icm.parameters()), lr=1e-3)
        self.avg_reward = 0
        self.avg_finish_cases = 0
        self.avg_step = 0
        self.model.to(self.device)
        self.tar_model = deepcopy(self.model)

        self.scheduler = MultiStepLR(self.optimizer, milestones=[200000, 400000], gamma=0.5)
        self.buffer = buffer

        self.done = False
        self.weights_id = None

        self.store_weights()

        self.learning_thread = None

        self.my_summary = summary

        self.counter = 0
        self.last_counter = 0
        self.loss = 0

        self.use_extrinsic = True
        self.td_loss_scale = 1
        self.forward_loss_scale = 0.5
        self.inverse_loss_scale = 0.5
        self.intrinsic_reward_scale = 5

    def get_weights(self):
        return self.weights_id

    def q_loss(self, b_obs, b_action, b_reward, b_done, b_hidden, epoch):
        batch_size = b_obs.shape[0]
        b_steps = torch.ones((batch_size, 1))
        # TODO://
        b_q = self.model(b_obs[:, :-configs.forward_steps], b_hidden).gather(1, b_action)
        with torch.no_grad():
            b_q_ = (1 - b_done) * self.tar_model(b_obs[:, -configs.forward_steps:], b_hidden).max(1, keepdim=True)[0]
        td_error = (b_q - (b_reward + (0.99 ** b_steps) * b_q_))
        loss = self.huber_loss(td_error).mean()

        self.my_summary.add_float.remote(x=epoch, y=loss.item(), title="TD Loss",
                                         x_name=f"trained epoch")

        return td_error, loss

    def compute_icm_loss(self, b_obs, b_action, b_reward, b_done, b_steps, b_seq_len, b_hidden, b_comm_mask,
                         idxes, weights, pre_obs, epoch, r_t):
        s_t = b_obs[torch.arange(b_obs.shape[0]), b_seq_len - 1]
        s_t_prime = b_obs[torch.arange(b_obs.shape[0]), b_seq_len]
        a_t = b_action
        a_vec = F.one_hot(a_t, num_classes=5).squeeze(1)
        prediction_s_next, prediction_a_vec, feature_x_next = self.icm.get_full(s_t, s_t_prime, a_vec)

        forward_loss = F.mse_loss(prediction_s_next, feature_x_next.detach(), reduction='none')
        inverse_prediction_loss = F.cross_entropy(prediction_a_vec, b_action.squeeze(1).detach(), reduction='none')

        total_rewards = forward_loss.mean(-1, keepdim=True).clone()
        if self.use_extrinsic:
            total_rewards += r_t

        self.my_summary.add_float.remote(x=epoch, y=forward_loss.mean().item(), title="Forward Loss",
                                         x_name=f"trained epoch")
        self.my_summary.add_float.remote(x=epoch, y=inverse_prediction_loss.mean().item(),
                                         title="Inverse Prediction Loss",
                                         x_name=f"trained epoch")
        icm_loss = self.forward_loss_scale * forward_loss.mean() + self.inverse_loss_scale * inverse_prediction_loss.mean()

        return icm_loss

    def store_weights(self):
        state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        self.weights_id = ray.put(state_dict)

    def run(self):
        self.learning_thread = threading.Thread(target=self.train, daemon=True)
        self.learning_thread.start()

    def get_data(self):
        data_id = ray.get(self.buffer.get_data.remote())
        data = ray.get(data_id)
        b_obs, b_action, b_reward, b_done, b_hidden, pos, goal = data

        b_obs, b_action, b_reward = b_obs.to(self.device), b_action.to(self.device), \
            b_reward.to(self.device)
        b_done = b_done.to(self.device),
        b_hidden = b_hidden.to(self.device)
        return b_obs, b_action, b_reward, b_done, b_hidden, pos, goal

    def param_update(self, loss, scaler):
        self.optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), 40)

        scaler.step(self.optimizer)
        scaler.update()

        self.scheduler.step()

    def draw(self, obs, pos, action, goal, reward):
        batch, seq_len, agent_num, _, _, _ = obs.size()
        for b in range(batch):
            pos_item = pos[b]
            goal_item = goal[b]
            with open("action.txt", 'w') as f:
                f.write(str(action[0]))
            for l in range(seq_len):
                pos_item_per_seq = pos_item[l][0]
                goal_item_per_seq = goal_item[l][0]
                obs_item = obs[b][l][0]
                agent_map = obs_item[0]
                obstacle_map = obs_item[1]
                top, down, left, right = obs_item[2:]
                prefix = f"batch_{b}_seq_{l}_"
                show_map = [(agent_map, 'agent_map.png'), (obstacle_map, "obstacle_map.png"),
                            (top, 'top.png'), (down, "down.png"),
                            (left, 'left.png'), (right, "right.png"), ]
                for i in range(len(show_map)):
                    map_data, name = show_map[i]
                    name = prefix + name
                    show_two_map(self.map, agent_map, pos_item_per_seq, goal_item_per_seq, is_show=False, name=name)

            assert False

        pass

    def train(self):
        scaler = GradScaler()
        epoch = 0
        while not ray.get(self.buffer.check_done.remote()):
            epoch += 1
            step_length = 10000
            for i in range(1, step_length):

                b_obs, b_action, b_reward, b_done, b_hidden, pos, goal = self.get_data()

                td_error, loss = self.q_loss(b_obs, b_action, b_reward, b_done, b_hidden, epoch)

                self.draw(b_obs, pos, b_action, goal, b_reward)
                # if i % 3 == 0:
                #     loss += self.compute_icm_loss(b_obs, b_action, b_reward, b_done, b_steps, b_seq_len, b_hidden,
                #                                   b_comm_mask, \
                #                                   idxes, weights, pre_obs, epoch, r_t)

                priorities = td_error.detach().squeeze().abs().clamp(1e-4).cpu().numpy()

                self.loss += loss.item()

                self.param_update(loss, scaler)

                if i % 5 == 0:
                    self.store_weights()

                # update target net, save model
                if i % configs.target_network_update_freq == 0:
                    self.tar_model.load_state_dict(self.model.state_dict())

                self.counter += 1
                if i % configs.save_interval == 0:
                    now_time = time.strftime("%Y-%m-%d-%H", time.localtime())
                    path = os.path.join(configs.save_path, '{}-{}.pth'.format(now_time,
                                                                              self.counter))
                    model_save(self.model, path)
                    print(
                        "save model path:" + os.path.join(configs.save_path, '{}-{}.pth'.
                                                          format(now_time, self.counter)))

        self.done = True

    def huber_loss(self, td_error, kappa=1.0):
        abs_td_error = td_error.abs()
        flag = (abs_td_error < kappa).float()
        return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)

    def stats(self, interval: int):
        print('number of updates: {}'.format(self.counter))
        print('update speed: {}/s'.format((self.counter - self.last_counter) / interval))
        if self.counter != self.last_counter:
            average_loss = self.loss / (self.counter - self.last_counter)
            print('loss: {:.4f}'.format(average_loss))
        self.last_counter = self.counter
        self.loss = 0
        return self.done
