import copy
from typing import List
import matplotlib.pyplot as plt
import numpy as np

import configs
from construct_map.static_map import StaticObstacle
from construct_map.dynamic_map import DynamicPedestrian
import pandas as pd
from utils.math_tool import *
import torch

plt.ion()

int_action = {0: "stay", 1: "up", 2: "down", 3: "left", 4: "right"}
action_list = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=int)
select_list = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=int)

color_map = np.array([[255, 255, 255],  # white
                      [190, 190, 190],  # gray
                      [0, 191, 255],  # blue
                      [255, 165, 0],  # orange
                      [0, 250, 154]])  # green


def map_partition(map):
    """
    partitioning map into independent partitions
    把可行区域提取出来
    """
    empty_pos = np.argwhere(map == 0).astype(int).tolist()

    empty_pos = [tuple(pos) for pos in empty_pos]

    if not empty_pos:
        raise RuntimeError('no empty position')

    partition_list = list()
    while empty_pos:

        start_pos = empty_pos.pop()

        open_list = list()
        open_list.append(start_pos)
        close_list = list()

        while open_list:
            x, y = open_list.pop(0)

            up = x - 1, y
            if up[0] >= 0 and map[up] == 0 and up in empty_pos:
                empty_pos.remove(up)
                open_list.append(up)

            down = x + 1, y
            if down[0] < map.shape[0] and map[down] == 0 and down in empty_pos:
                empty_pos.remove(down)
                open_list.append(down)

            left = x, y - 1
            if left[1] >= 0 and map[left] == 0 and left in empty_pos:
                empty_pos.remove(left)
                open_list.append(left)

            right = x, y + 1
            if right[1] < map.shape[1] and map[right] == 0 and right in empty_pos:
                empty_pos.remove(right)
                open_list.append(right)

            close_list.append((x, y))

        partition_list.append(close_list)

    return partition_list


class Environment:

    def _normal_generate(self):
        partition_list = copy.deepcopy(self.partition_list)
        self.agents_pos = np.empty((self.num_agents, 2), dtype=int)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=int)
        pos_num = sum([len(partition) for partition in partition_list])

        # loop to assign agent original position and goal position for each agent
        for i in range(self.num_agents):

            pos_idx = random.randint(0, pos_num - 1)
            partition_idx = 0
            for partition in partition_list:
                if pos_idx >= len(partition):
                    pos_idx -= len(partition)
                    partition_idx += 1
                else:
                    break

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.agents_pos[i] = np.asarray(pos, dtype=int)

            x, y, is_in = grid2coord(self.agents_pos[i])
            if not is_in:
                assert is_in, 'agent id = {}, start position = {}, is in Plane = {}'.format(i, [x, y], is_in)
            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=int)

            partition_list = [partition for partition in partition_list if len(partition) >= 2]
            pos_num = sum([len(partition) for partition in partition_list])

    def _random_generate(self):
        agent_list, blank_list = self.generate_n_position(4)
        generate_list = self.generate_length_n(blank_list, agent_list, self.distance)
        self.agents_pos = np.array(agent_list, dtype=int)
        self.goals_pos = np.array(generate_list, dtype=int)

    def generate_agent_and_goal(self):
        if self.use_random:
            self._random_generate()
        else:
            self._normal_generate()

    def __init__(self, num_agents: int = configs.init_env_settings[0], map_size: tuple = configs.map_size,
                 map_length: int = configs.init_env_settings[1],
                 obs_radius: int = configs.obs_radius, reward_fn: dict = configs.reward_fn, fix_density=None,
                 curriculum=False, init_env_settings_set=configs.init_env_settings):
        self.heuri_map = None
        self.dist_map = None
        self.distance = 1

        self.fig = None
        self.dyn_map = 0
        self.curriculum = curriculum
        self.use_random = False
        if curriculum:
            self.env_set = [init_env_settings_set]
            self.num_agents = init_env_settings_set[0]
            self.map_size = map_size
        else:
            self.num_agents = num_agents
            self.map_size = map_size

        self.static_obs = StaticObstacle()
        self.map = self.static_obs.static_map
        self.dynamic_ped = DynamicPedestrian(self.static_obs.rows, self.static_obs.columns)
        self.t = 0

        partition_list = map_partition(self.map)
        partition_list = [partition for partition in partition_list if len(partition) >= 2]

        self.partition_list = partition_list
        self.first_blank_list = partition_list[0]

        self.generate_agent_and_goal()

        self.obs_radius = obs_radius

        self.reward_fn = reward_fn
        self.get_heuri_map()
        self.steps = 0
        self.last_actions = np.zeros((self.num_agents, 5, 2 * obs_radius + 1, 2 * obs_radius + 1), dtype=bool)
        self.imgs = []

    def set_distance(self, distance):
        self.distance = distance

    def update_env_settings_set(self, new_env_settings_set):
        self.env_set = new_env_settings_set

    def reset(self, num_agents=None, map_size: tuple = configs.map_size, map_length=None):

        if self.curriculum:
            rand = random.choice(self.env_set)
            self.num_agents = rand[0]
            self.map_size = map_size

        elif num_agents is not None and map_length is not None:
            self.num_agents = num_agents
            self.map_size = map_size

        ''''地图设置，根据unity导出的数据构建静态地图'''
        self.static_obs = StaticObstacle()
        self.map = self.static_obs.static_map
        self.dynamic_ped = DynamicPedestrian(self.static_obs.rows, self.static_obs.columns)
        self.t = 1

        partition_list = map_partition(self.map)
        partition_list = [partition for partition in partition_list if len(partition) >= 2]

        assert self.partition_list == partition_list

        self.generate_agent_and_goal()

        self.steps = 0
        self.get_heuri_map()

        self.last_actions = np.zeros((self.num_agents, 5, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
                                     dtype=bool)

        return self.observe()

    def load(self, map: np.ndarray, agents_pos: np.ndarray, goals_pos: np.ndarray):

        self.map = np.copy(map)
        self.agents_pos = np.copy(agents_pos)
        self.goals_pos = np.copy(goals_pos)

        self.num_agents = agents_pos.shape[0]
        self.map_size = (self.map.shape[0], self.map.shape[1])

        self.steps = 0

        self.imgs = []

        self.get_heuri_map()

        self.last_actions = np.zeros((self.num_agents, 5, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
                                     dtype=bool)

    def get_heuri_map(self):
        # 每个agent都有
        dist_map = np.ones((self.num_agents, *self.map_size), dtype=np.int32) * 2147483647
        for i in range(self.num_agents):
            open_list = list()
            x, y = tuple(self.goals_pos[i])
            open_list.append((x, y))
            dist_map[i, x, y] = 0

            while open_list:
                x, y = open_list.pop(0)
                dist = dist_map[i, x, y]

                up = x - 1, y
                if up[0] >= 0 and self.map[up] == 0 and dist_map[i, x - 1, y] > dist + 1:
                    dist_map[i, x - 1, y] = dist + 1
                    if up not in open_list:
                        open_list.append(up)

                down = x + 1, y
                if down[0] < self.map_size[0] and self.map[down] == 0 and dist_map[i, x + 1, y] > dist + 1:
                    dist_map[i, x + 1, y] = dist + 1
                    if down not in open_list:
                        open_list.append(down)

                left = x, y - 1
                if left[1] >= 0 and self.map[left] == 0 and dist_map[i, x, y - 1] > dist + 1:
                    dist_map[i, x, y - 1] = dist + 1
                    if left not in open_list:
                        open_list.append(left)

                right = x, y + 1
                if right[1] < self.map_size[1] and self.map[right] == 0 and dist_map[i, x, y + 1] > dist + 1:
                    dist_map[i, x, y + 1] = dist + 1
                    if right not in open_list:
                        open_list.append(right)
        self.dist_map = dist_map
        self.heuri_map = np.zeros((self.num_agents, 4, *self.map_size), dtype=bool)

        for x in range(self.map_size[0]):
            for y in range(self.map_size[1]):
                if self.map[x, y] == 0:  # 可行点
                    for i in range(self.num_agents):
                        # 往上走
                        if x > 0 and dist_map[i, x - 1, y] < dist_map[i, x, y]:
                            assert dist_map[i, x - 1, y] == dist_map[i, x, y] - 1
                            self.heuri_map[i, 0, x, y] = 1
                        # 往下走
                        if x < self.map_size[0] - 1 and dist_map[i, x + 1, y] < dist_map[i, x, y]:
                            assert dist_map[i, x + 1, y] == dist_map[i, x, y] - 1
                            self.heuri_map[i, 1, x, y] = 1
                        # 往左走
                        if y > 0 and dist_map[i, x, y - 1] < dist_map[i, x, y]:
                            assert dist_map[i, x, y - 1] == dist_map[i, x, y] - 1
                            self.heuri_map[i, 2, x, y] = 1
                        # 往右走
                        if y < self.map_size[1] - 1 and dist_map[i, x, y + 1] < dist_map[i, x, y]:
                            assert dist_map[i, x, y + 1] == dist_map[i, x, y] - 1
                            self.heuri_map[i, 3, x, y] = 1
        # 填充;在各维度的各个方向上想要填补的长度
        self.heuri_map = np.pad(self.heuri_map, (
            (0, 0), (0, 0), (self.obs_radius, self.obs_radius), (self.obs_radius, self.obs_radius)))

    def generate_length_n(self, blank_area_list, agent_position_list, distance):
        blank_area_list = copy.deepcopy(blank_area_list)
        goal_generate_list = []
        for i in range(configs.num_agents):
            record_map = set()
            test_time = 0
            while True:
                test_time += 1
                if len(record_map) == len(select_list):
                    temp = random.randint(0, len(blank_area_list) - 1)
                    goal_generate_list.append(list(blank_area_list)[temp])
                    blank_area_list.remove(list(blank_area_list)[temp])
                    break
                direction = random.randint(0, len(select_list) - 1)
                agent_position = agent_position_list[i]
                action = select_list[direction]
                selected_x, selected_y = agent_position[0] + distance * action[0], \
                                         agent_position[1] + distance * action[1]
                if selected_x < 0 or selected_y < 0 or selected_x >= self.map.shape[0] \
                        or selected_y >= self.map.shape[1]:
                    record_map.add(direction)
                    continue
                if self.map[selected_x, selected_y] == 1:
                    record_map.add(direction)
                    continue
                if (selected_x, selected_y) not in blank_area_list:
                    record_map.add(direction)
                    continue
                goal_generate_list.append((selected_x, selected_y))
                blank_area_list.remove((selected_x, selected_y))
                break
        return goal_generate_list

    def generate_n_position(self, n):
        agent_list = []
        blank_list_copy = copy.deepcopy(self.first_blank_list)
        assert len(self.first_blank_list) >= n
        list_n = len(self.first_blank_list)
        while len(agent_list) != n:
            idx = random.randint(0, list_n - 1)
            agent_list.append(blank_list_copy[idx])
            del blank_list_copy[idx]
            list_n = len(blank_list_copy)
        return agent_list, set(blank_list_copy)

    def step(self, actions: List[int], pde_df=0):
        '''
        actions:
            list of indices
                0 stay
                1 up
                2 down
                3 left
                4 right
        '''
        rank_of_agents = rank_agent_by_distance(self.agents_pos, self.goals_pos)
        self.dyn_map = pde_df
        assert len(actions) == self.num_agents, 'only {} actions as input while {} agents in environment'.format(
            len(actions), self.num_agents)
        assert all([5 > action_idx >= 0 for action_idx in actions]), 'action index out of range'

        checking_list = [i for i in range(self.num_agents)]

        rewards = []
        next_pos = np.copy(self.agents_pos)
        origin_pos = np.copy(self.agents_pos) + configs.obs_radius

        # remove unmoving agent id
        for agent_id in checking_list.copy():
            if actions[agent_id] == 0:
                # unmoving

                if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                    rewards.append(self.reward_fn['stay_on_goal'])
                else:
                    rewards.append(self.reward_fn['stay_off_goal'])

                checking_list.remove(agent_id)
            else:
                next_pos[agent_id] += action_list[actions[agent_id]]
                rewards.append(self.reward_fn['move'])

        # first round check, these two conflicts have the heightest priority
        for agent_id in checking_list.copy():

            if np.any(next_pos[agent_id] < 0) or (next_pos[agent_id][0] >= self.map_size[0]) or (
                    next_pos[agent_id][1] >= self.map_size[1]):
                # if np.any(next_pos[agent_id] < 0) or np.any(next_pos[agent_id] >= self.map_size[0]):
                # agent out of map range
                rewards[agent_id] = self.reward_fn['collision']
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)
            elif self.map[tuple(next_pos[agent_id])] == 1:
                # collide obstacle
                rewards[agent_id] = self.reward_fn['collision']
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)

        # second round check, agent swapping conflict
        no_conflict = False
        while not no_conflict:

            no_conflict = True
            for agent_id in checking_list.copy():

                target_agent_id = np.where(np.all(next_pos[agent_id] == self.agents_pos, axis=1))[0]

                if len(target_agent_id) == 1:

                    target_agent_id = target_agent_id.item()
                    assert target_agent_id != agent_id, 'logic bug'

                    if np.array_equal(next_pos[target_agent_id], self.agents_pos[agent_id]):
                        assert target_agent_id in checking_list, 'target_agent_id should be in checking list'

                        next_pos[agent_id] = self.agents_pos[agent_id]
                        rewards[agent_id] = self.reward_fn['collision']

                        next_pos[target_agent_id] = self.agents_pos[target_agent_id]
                        rewards[target_agent_id] = self.reward_fn['collision']

                        checking_list.remove(agent_id)
                        checking_list.remove(target_agent_id)

                        no_conflict = False
                        break
                else:
                    content = 'len(target_agent_id) != 1 and len(target_agent_id) == {}'.format(len(target_agent_id))

        # third round check, agent collision conflict
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list.copy():

                collide_agent_id = np.where(np.all(next_pos == next_pos[agent_id], axis=1))[0].tolist()
                if len(collide_agent_id) > 1:
                    # collide agent

                    # if all agents in collide agent are in checking list
                    all_in_checking = True
                    for id in collide_agent_id.copy():
                        if id not in checking_list:
                            all_in_checking = False
                            collide_agent_id.remove(id)

                    if all_in_checking:

                        collide_agent_pos = next_pos[collide_agent_id].tolist()
                        for pos, id in zip(collide_agent_pos, collide_agent_id):
                            pos.append(id)
                        collide_agent_pos.sort(key=lambda x: x[0] * self.map_size[0] + x[1])

                        collide_agent_id.remove(collide_agent_pos[0][2])

                    next_pos[collide_agent_id] = self.agents_pos[collide_agent_id]
                    for id in collide_agent_id:
                        rewards[id] = self.reward_fn['collision']

                    for id in collide_agent_id:
                        checking_list.remove(id)

                    no_conflict = False
                    break

        self.agents_pos = np.copy(next_pos)

        self.steps += 1

        # check done
        if np.array_equal(self.agents_pos, self.goals_pos):
            done = True
            rewards = [self.reward_fn['finish'] for _ in range(self.num_agents)]
        else:
            done = False

        reward_signal = [x.item() for x in torch.from_numpy((self.agents_pos == self.goals_pos)).sum(dim=1)]
        rewards = [self.reward_fn['finish'] if reward_signal[i] > 1 else rewards[i] for i in range(self.num_agents)]

        info = {'step': self.steps - 1}

        # make sure no overlapping agents
        if np.unique(self.agents_pos, axis=0).shape[0] < self.num_agents:
            print(self.steps)
            print(self.map)
            print(self.agents_pos)
            raise RuntimeError('unique')

        # update last actions
        self.last_actions = np.zeros((self.num_agents, 5, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
                                     dtype=bool)
        self.last_actions[np.arange(self.num_agents), np.array(actions)] = 1

        c_a = actions[0] - 1
        obs = self.observe()
        #
        if done:
            rewards = [1]
        else:
            if c_a == -1:
                rewards = [-2]
            else:
                temp = origin_pos[0]
                rewards = [1] if self.heuri_map[0][c_a][temp[0]][temp[1]].item() else [-2]

        return obs, rewards, done, info

    def observe(self):
        """
        return observation and position for each agent

        obs: shape (num_agents, 11, 2*obs_radius+1, 2*obs_radius+1)
            layer 1: agent map
            layer 2: obstacle map
            layer 3-6: heuristic map
            layer 7-11: one-hot representation of agent's last action

        pos: used for caculating communication mask

        """
        if isinstance(self.dyn_map, pd.DataFrame) and self.dyn_map:  # 数据类型判断
            dynamic_ped_map = self.dyn_map
        else:
            coorlist = self.dynamic_ped.pde_df.iloc[self.t].tolist()
            dynamic_ped_map = self.dynamic_ped.get_pedcoor(coorlist)
            self.t = self.t + 1
            # todo 只是测试，之后删掉
            if self.t >= len(self.dynamic_ped.pde_df):
                self.t = 0
        self.map = self.static_obs.static_map
        self.map[np.where(self.map >= 1)] = 1

        obs = np.zeros((self.num_agents, 6, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1), dtype=bool)

        # 0 represents obstacle to match 0 padding in CNN 地图上下左右都进行填充
        obstacle_map = np.pad(self.map, self.obs_radius, 'constant', constant_values=0)  # 对边缘进行填充

        agent_map = np.zeros((self.map_size), dtype=bool)
        agent_map[self.agents_pos[:, 0], self.agents_pos[:, 1]] = 1
        agent_map = np.pad(agent_map, self.obs_radius, 'constant', constant_values=0)

        for i, agent_pos in enumerate(self.agents_pos):
            x, y = agent_pos

            obs[i, 0] = agent_map[x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]
            obs[i, 0, self.obs_radius, self.obs_radius] = 0
            obs[i, 1] = obstacle_map[x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1].copy()
            obs[i, 2:] = self.heuri_map[i, :, x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]

        return obs, np.copy(self.agents_pos), np.copy(self.goals_pos)

    def render(self, action):
        if not hasattr(self, 'fig'):
            self.fig = plt.figure()

        map = np.copy(self.map)
        for agent_id in range(self.num_agents):
            if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                map[tuple(self.agents_pos[agent_id])] = 4
            else:
                map[tuple(self.agents_pos[agent_id])] = 2
                map[tuple(self.goals_pos[agent_id])] = 3

        map = map.astype(np.uint8)

        # add text in plot
        self.imgs.append([])
        if hasattr(self, 'texts'):
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(self.agents_pos, self.goals_pos)):
                info = int_action[action[i]]
                self.texts[i][0].set_position((agent_y, agent_x))
                self.texts[i][0].set_text(i)

                self.texts[i][1].set_position((goal_y, goal_x))
                self.texts[i][1].set_text(i)

                self.texts[i][2].set_position((agent_y + 1, agent_x + 1))
                self.texts[i][2].set_text(info)


        else:
            self.texts = []
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(self.agents_pos, self.goals_pos)):
                info = int_action[action[i]]
                agent_text = plt.text(agent_y, agent_x, i, color='black', ha='center', va='center')
                goal_text = plt.text(goal_y, goal_x, i, color='black', ha='center', va='center')
                action_text = plt.text(agent_y + 1, agent_x + 1, info, color='black', ha='center', va='center')
                self.texts.append((agent_text, goal_text, action_text))

        plt.imshow(color_map[map], animated=True)

        plt.show()
        plt.pause(0.5)

    def close(self, save=False):
        plt.close()
        del self.fig


if __name__ == '__main__':
    e = Environment()
    e.generate_agent_and_goal()
