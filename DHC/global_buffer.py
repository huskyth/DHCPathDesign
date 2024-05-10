import threading
import time
from typing import Tuple
import ray
import numpy as np
import configs
import torch


@ray.remote(num_cpus=1)
class GlobalBuffer:
    def __init__(self, episode_capacity=configs.episode_capacity,
                 local_buffer_capacity=configs.max_episode_length,
                 init_env_settings=configs.init_env_settings,
                 alpha=configs.prioritized_replay_alpha, beta=configs.prioritized_replay_beta):

        self.capacity = episode_capacity
        self.local_buffer_capacity = local_buffer_capacity
        self.size = 0
        self.ptr = 0

        # prioritized experience replay
        self.alpha = alpha
        self.beta = beta

        self.counter = 0
        self.batched_data = []
        self.stat_dict = {init_env_settings: []}
        self.lock = threading.Lock()
        self.env_settings_set = ray.put([init_env_settings])

        self.obs_buf = np.zeros(
            ((local_buffer_capacity + 1) * episode_capacity, configs.max_num_agents, *configs.obs_shape), dtype=bool)
        self.pos_buf = np.zeros(
            ((local_buffer_capacity + 1) * episode_capacity, configs.max_num_agents, 2), dtype=int)
        self.pre_obs_buf = np.zeros(
            ((local_buffer_capacity + 1) * episode_capacity, configs.max_num_agents, *configs.obs_shape), dtype=bool)
        self.act_buf = np.zeros((local_buffer_capacity * episode_capacity), dtype=np.uint8)
        self.rew_buf = np.zeros((local_buffer_capacity * episode_capacity), dtype=np.float16)
        self.hid_buf = np.zeros((local_buffer_capacity * episode_capacity, configs.max_num_agents, configs.hidden_dim),
                                dtype=np.float16)
        self.done_buf = np.zeros(episode_capacity, dtype=bool)
        self.size_buf = np.zeros(episode_capacity, dtype=np.uint)
        self.comm_mask_buf = np.zeros(
            ((local_buffer_capacity + 1) * episode_capacity, configs.max_num_agents, configs.max_num_agents),
            dtype=bool)

        self.background_thread = None

    def __len__(self):
        return self.size

    def run(self):
        self.background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        self.background_thread.start()

    def prepare_data(self):
        while True:
            if len(self.batched_data) <= 4:
                data = self.sample_batch(configs.batch_size)
                data_id = ray.put(data)
                self.batched_data.append(data_id)
            else:
                time.sleep(0.1)

    def get_data(self):
        if len(self.batched_data) == 0:
            print('no prepared data')
            data = self.sample_batch(configs.batch_size)
            data_id = ray.put(data)
            return data_id
        else:
            return self.batched_data.pop(0)

    def add(self, data: Tuple):
        '''
        data: actor_id 0, num_agents 1, map_len 2, obs_buf 3, act_buf 4, rew_buf 5,
         hid_buf 6, td_errors 7, done 8, size 9, return value 10, pre_obs_buf 11, position_buf 12
        '''
        if data[0] >= 12:
            stat_key = (data[1], data[2])
            if stat_key in self.stat_dict:
                self.stat_dict[stat_key].append(data[8])
                if len(self.stat_dict[stat_key]) == 201:
                    self.stat_dict[stat_key].pop(0)

        with self.lock:
            idxes = np.arange(self.ptr * self.local_buffer_capacity,
                              (self.ptr + 1) * self.local_buffer_capacity)
            start_idx = self.ptr * self.local_buffer_capacity
            # update buffer size
            self.size -= self.size_buf[self.ptr].item()
            self.size += data[9]
            self.counter += data[9]

            self.obs_buf[start_idx + self.ptr:start_idx + self.ptr + data[9] + 1, :data[1]] = data[3]
            self.pos_buf[start_idx + self.ptr:start_idx + self.ptr + data[9] + 1, :data[1], :] = data[13]
            self.pre_obs_buf[start_idx + self.ptr:start_idx + self.ptr + data[9] + 1, :data[1]] = data[12]
            self.act_buf[start_idx:start_idx + data[9]] = data[4]
            self.rew_buf[start_idx:start_idx + data[9]] = data[5]
            self.hid_buf[start_idx:start_idx + data[9], :data[1]] = data[6]
            self.done_buf[self.ptr] = data[8]
            self.size_buf[self.ptr] = data[9]

            self.ptr = (self.ptr + 1) % self.capacity

    def random_sample(self, batch_size):
        ret = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            # TODO://check ptr
            selected = np.random.randint(0, 1 if self.ptr == 0 else self.ptr)
            ret[i] = np.random.randint(0, self.size_buf[selected]) + selected * self.local_buffer_capacity
        return ret

    def sample_batch(self, batch_size: int) -> Tuple:
        b_obs, b_action, b_reward, b_done, b_steps = [], [], [], [], []
        b_hidden = []
        b_pre_obs = []
        r_t = []
        pos_list = []
        with self.lock:
            idxes = self.random_sample(batch_size)
            global_idxes = idxes // self.local_buffer_capacity
            local_idxes = idxes % self.local_buffer_capacity

            for idx, global_idx, local_idx in zip(idxes.tolist(), global_idxes.tolist(),
                                                  local_idxes.tolist()):

                assert local_idx < self.size_buf[global_idx], 'index is {} but size is {}' \
                    .format(local_idx, self.size_buf[global_idx])

                steps = min(configs.forward_steps, (self.size_buf[global_idx].item() - local_idx))

                if local_idx < configs.seq_len - 1:
                    obs = self.obs_buf[global_idx * (self.local_buffer_capacity + 1):
                                       idx + global_idx + 1 + steps]
                    pos = self.pos_buf[global_idx * (self.local_buffer_capacity + 1):
                                       idx + global_idx + 1 + steps]
                    pre_obs = self.pre_obs_buf[global_idx * (self.local_buffer_capacity + 1):
                                               idx + global_idx + 1 + steps]
                    hidden = np.zeros((configs.max_num_agents, configs.hidden_dim), dtype=np.float16)
                elif local_idx == configs.seq_len - 1:
                    obs = self.obs_buf[idx + global_idx + 1 - configs.seq_len:
                                       idx + global_idx + 1 + steps]
                    pos = self.pos_buf[idx + global_idx + 1 - configs.seq_len:
                                       idx + global_idx + 1 + steps]
                    pre_obs = self.pre_obs_buf[idx + global_idx + 1 - configs.seq_len:
                                               idx + global_idx + 1 + steps]
                    hidden = np.zeros((configs.max_num_agents, configs.hidden_dim), dtype=np.float16)
                else:
                    obs = self.obs_buf[idx + global_idx + 1 - configs.seq_len:
                                       idx + global_idx + 1 + steps]
                    pos = self.pos_buf[idx + global_idx + 1 - configs.seq_len:
                                       idx + global_idx + 1 + steps]
                    pre_obs = self.pre_obs_buf[idx + global_idx + 1 - configs.seq_len:
                                               idx + global_idx + 1 + steps]
                    hidden = self.hid_buf[idx - configs.seq_len]

                if obs.shape[0] < configs.seq_len + configs.forward_steps:
                    pad_len = configs.seq_len + configs.forward_steps - obs.shape[0]
                    obs = np.pad(obs, ((0, pad_len), (0, 0), (0, 0), (0, 0), (0, 0)))
                if pre_obs.shape[0] < configs.seq_len + configs.forward_steps:
                    pre_obs = np.pad(pre_obs, ((0, pad_len), (0, 0), (0, 0), (0, 0), (0, 0)))
                action = self.act_buf[idx]
                r_t_value = self.rew_buf[idx]
                reward = 0
                for i in range(steps):
                    reward += self.rew_buf[idx + i] * 0.99 ** i

                if self.done_buf[global_idx] and local_idx >= self.size_buf[global_idx] \
                        - configs.forward_steps:
                    done = True
                else:
                    done = False

                b_obs.append(obs)
                b_pre_obs.append(pre_obs)
                b_action.append(action)
                b_reward.append(reward)
                b_done.append(done)
                b_steps.append(steps)
                b_hidden.append(hidden)
                r_t.append(r_t_value)
                pos_list.append(pos)

            data = (
                torch.from_numpy(np.stack(b_obs).astype(np.float16)),
                torch.LongTensor(b_action).unsqueeze(1),
                torch.HalfTensor(b_reward).unsqueeze(1),
                torch.HalfTensor(b_done).unsqueeze(1),
                torch.HalfTensor(b_steps).unsqueeze(1),
                torch.from_numpy(np.concatenate(b_hidden)),

                idxes,
                self.ptr,
                torch.from_numpy(np.stack(b_pre_obs).astype(np.float16)),
                torch.HalfTensor(r_t).unsqueeze(1),
                pos_list
            )
            return data

    def log(self, interval, use=False):
        if not use: return
        print(60 * '*')
        print('buffer update speed: {}/s'.format(self.counter / interval))
        print('buffer size: {}'.format(self.size))

        print('  ', end='')
        for i in range(configs.init_env_settings[1], configs.max_map_lenght + 1, 5):
            print('   {:2d}   '.format(i), end='')
        print()

        for num_agents in range(configs.init_env_settings[0], configs.max_num_agents + 1):
            print('{:2d}'.format(num_agents), end='')
            for map_len in range(configs.init_env_settings[1], configs.max_map_lenght + 1, 5):
                if (num_agents, map_len) in self.stat_dict:
                    print('{:4d}/{:<3d}'.format(sum(self.stat_dict[(num_agents, map_len)]),
                                                len(self.stat_dict[(num_agents, map_len)])), end='')
                else:
                    print('   N/A  ', end='')
            print()

    def stats(self, interval: int):
        self.log(interval, use=True)
        for key, val in self.stat_dict.copy().items():
            if len(val) == 200 and sum(val) >= 200 * configs.pass_rate:
                # add number of agents
                add_agent_key = (key[0] + 1, key[1])
                if add_agent_key[0] <= configs.max_num_agents and add_agent_key not in self.stat_dict:
                    self.stat_dict[add_agent_key] = []
                if key[1] < configs.max_map_lenght:
                    add_map_key = (key[0], key[1] + 5)
                    if add_map_key not in self.stat_dict:
                        self.stat_dict[add_map_key] = []
        self.env_settings_set = ray.put(list(self.stat_dict.keys()))
        self.counter = 0

    def ready(self):
        if len(self) >= configs.learning_starts:
            return True
        else:
            return False

    def get_env_settings(self):
        return self.env_settings_set

    def check_done(self):
        for i in range(configs.max_num_agents):
            if (i + 1, configs.max_map_lenght) not in self.stat_dict:
                return False
            l = self.stat_dict[(i + 1, configs.max_map_lenght)]
            if len(l) < 200:
                return False
            elif sum(l) < 200 * configs.pass_rate:
                return False
        return True
