import random
from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
from matplotlib import colors
import configs
from construct_map.static_map import StaticObstacle
from construct_map.dynamic_map import DynamicPedestrian
from utils.logger_write_file import *
import pandas as pd

'''
我修改了，原代码也是这样
actions:
    list of indices
        0 stay
        1 up
        2 down
        3 left
        4 right
'''
action_list = np.array([[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0]], dtype=np.int)

color_map = np.array([[255, 255, 255],  # white
                      [190, 190, 190],  # gray
                      [0, 191, 255],  # blue
                      [255, 165, 0],  # orange
                      [0, 250, 154]])  # green


# print("dynamic map")

def map_partition(map):
    """
    partitioning map into independent partitions
    把可行区域提取出来
    """
    # 值为0的数组的索引；即把free区域的位置坐标取出来
    empty_pos = np.argwhere(map == 0).astype(np.int).tolist()

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


from utils.coordinate_tool import *
from utils.math_tool import *

class Environment:
    def __init__(self, num_agents: int = configs.init_env_settings[0], map_size: tuple = configs.map_size,
                 map_length: int = configs.init_env_settings[1],
                 obs_radius: int = configs.obs_radius, reward_fn: dict = configs.reward_fn, fix_density=None,
                 curriculum=False, init_env_settings_set=configs.init_env_settings):
        self.dyn_map = 0
        self.curriculum = curriculum
        if curriculum:
            self.env_set = [init_env_settings_set]
            self.num_agents = init_env_settings_set[0]
            # self.map_size = (init_env_settings_set[1], init_env_settings_set[1])
            self.map_size = map_size
        else:
            self.num_agents = num_agents
            # self.map_size = (map_length, map_length)
            self.map_size = map_size

        ''''地图设置，根据unity导出的数据构建静态地图'''
        # # set as same as in PRIMAL 设置
        # 根据给定的障碍密度和地图大小生成地图
        # if fix_density is None:
        #     self.fix_density = False
        #     self.obstacle_density = np.random.triangular(0, 0.33, 0.5)
        # else:
        #     self.fix_density = True
        #     self.obstacle_density = fix_density
        # # 从0-1序列中按照概率p采样map_size个数据；0代表free区域，1代表obstacle区域
        # self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.int)

        # 实例化静态地图类
        self.static_obs = StaticObstacle()
        self.map = self.static_obs.static_map  # 静态地图矩阵
        # 实例化动态地图类，step加入动态地图
        self.dynamic_ped = DynamicPedestrian(self.static_obs.rows, self.static_obs.columns)
        # 记录当前时刻
        self.t = 0

        partition_list = map_partition(self.map)
        partition_list = [partition for partition in partition_list if len(partition) >= 2]

        # 没有可行区域了，重新根据density等参数生成地图
        # while len(partition_list) == 0:
        #     self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.int)
        #     partition_list = map_partition(self.map)
        #     partition_list = [ partition for partition in partition_list if len(partition) >= 2 ]

        # 创建agent初始位置和目标位置数组
        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=np.int)
        # 可行区域的数量
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
            self.agents_pos[i] = np.asarray(pos, dtype=np.int)

            # 打印初始位置
            x, y, is_in = grid2coord(self.agents_pos[i])
            # print('agent id = {}, start position = {}, is in Plane = {}'.format(i, [x, y], is_in))
            if not is_in:
                assert is_in, 'agent id = {}, start position = {}, is in Plane = {}'.format(i, [x, y], is_in)
            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=np.int)

            partition_list = [partition for partition in partition_list if len(partition) >= 2]
            pos_num = sum([len(partition) for partition in partition_list])

        self.obs_radius = obs_radius

        self.reward_fn = reward_fn
        self.get_heuri_map()  # 静态地图
        self.steps = 0
        # 5应该是action空间
        self.last_actions = np.zeros((self.num_agents, 5, 2 * obs_radius + 1, 2 * obs_radius + 1), dtype=np.bool)

    def update_env_settings_set(self, new_env_settings_set):
        self.env_set = new_env_settings_set

    def reset(self, num_agents=None, map_size: tuple = configs.map_size, map_length=None):

        if self.curriculum:
            rand = random.choice(self.env_set)
            self.num_agents = rand[0]
            # self.map_size = (rand[1], rand[1])
            self.map_size = map_size  # 使用定义好的 mapsize

        elif num_agents is not None and map_length is not None:
            self.num_agents = num_agents
            # self.map_size = (map_length, map_length)
            self.map_size = map_size

        # 设定密度
        # if not self.fix_density:
        #     self.obstacle_density = np.random.triangular(0, 0.33, 0.5)

        # self.map = np.random.choice(2, self.map_size, p=[1-self.obstacle_density, self.obstacle_density]).astype(np.float32)

        ''''地图设置，根据unity导出的数据构建静态地图'''
        # set as same as in PRIMAL 设置
        # 实例化静态地图类
        self.static_obs = StaticObstacle()
        self.map = self.static_obs.static_map  # 静态地图矩阵
        # 实例化动态地图类
        self.dynamic_ped = DynamicPedestrian(self.static_obs.rows, self.static_obs.columns)
        # 记录当前时刻
        self.t = 1

        partition_list = map_partition(self.map)
        partition_list = [partition for partition in partition_list if len(partition) >= 2]

        self.agents_pos = np.empty((self.num_agents, 2), dtype=np.int)
        self.goals_pos = np.empty((self.num_agents, 2), dtype=np.int)

        pos_num = sum([len(partition) for partition in partition_list])

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
            self.agents_pos[i] = np.asarray(pos, dtype=np.int)

            pos = random.choice(partition_list[partition_idx])
            partition_list[partition_idx].remove(pos)
            self.goals_pos[i] = np.asarray(pos, dtype=np.int)

            partition_list = [partition for partition in partition_list if len(partition) >= 2]
            pos_num = sum([len(partition) for partition in partition_list])

        self.steps = 0
        self.get_heuri_map()

        self.last_actions = np.zeros((self.num_agents, 5, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
                                     dtype=np.bool)

        return self.observe()

    # 根据参数加载模型
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
                                     dtype=np.bool)

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

        self.heuri_map = np.zeros((self.num_agents, 4, *self.map_size), dtype=np.bool)

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

    # reward要受到动态行人影响
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
        assert all([action_idx < 5 and action_idx >= 0 for action_idx in actions]), 'action index out of range'

        checking_list = [i for i in range(self.num_agents)]

        rewards = []
        next_pos = np.copy(self.agents_pos)

        # remove unmoving agent id
        for agent_id in checking_list.copy():
            if actions[agent_id] == 0:
                # unmoving

                if np.array_equal(self.agents_pos[agent_id], self.goals_pos[agent_id]):
                    rewards.append(self.reward_fn['stay_on_goal'])
                else:
                    rewards.append(self.reward_fn['collision'] * rank_of_agents[agent_id] / len(self.goals_pos))
                    content = 'agent {} reward = {}, rank = {}'.format(agent_id, rewards[-1], rank_of_agents[agent_id])
                    write_file(content, 'rank_reward.txt')

                checking_list.remove(agent_id)
            else:
                # move 这个地方，超出
                origin_pos = [x for x in next_pos[agent_id]]
                next_pos[agent_id] += action_list[actions[agent_id]]
                # 添加朝着目标前进的奖赏为正，相反为负
                move_reward_sign = 1 if is_next_position_toward_goal_by_distance(*origin_pos, *next_pos[agent_id],
                                                                                 *list(
                                                                                     self.goals_pos[agent_id])) else -1
                rewards.append(abs(self.reward_fn['collision']) * move_reward_sign * rank_of_agents[agent_id] / len(self.agents_pos))
                content = 'agent {} reward = {}, rank = {}'.format(agent_id, rewards[-1], rank_of_agents[agent_id])
                write_file(content, 'rank_reward.txt')
                # print('start = {}, next = {}, goal = {}, reward_sign = {}, and reward = {}'.format(origin_pos,
                #                                                                                    next_pos[agent_id],
                #                                                                                    self.goals_pos[
                #                                                                                        agent_id],
                #                                                                                    move_reward_sign,
                #                                                                                    rewards[-1]))
        # first round check, these two conflicts have the heightest priority
        for agent_id in checking_list.copy():

            # 超出地图限制
            if np.any(next_pos[agent_id] < 0) or (next_pos[agent_id][0] >= self.map_size[0]) or (
                    next_pos[agent_id][1] >= self.map_size[1]):
                # if np.any(next_pos[agent_id] < 0) or np.any(next_pos[agent_id] >= self.map_size[0]):
                # agent out of map range
                rewards[agent_id] = self.reward_fn['collision']
                next_pos[agent_id] = self.agents_pos[agent_id]
                checking_list.remove(agent_id)
            # 碰到障碍物
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
                    # write_file(content)

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

                        # checking_list.remove(collide_agent_pos[0][2])

                    next_pos[collide_agent_id] = self.agents_pos[collide_agent_id]
                    for id in collide_agent_id:
                        rewards[id] = self.reward_fn['collision']

                    for id in collide_agent_id:
                        checking_list.remove(id)

                    no_conflict = False
                    break

        # self.history.append(np.copy(next_pos))
        self.agents_pos = np.copy(next_pos)

        self.steps += 1

        for i, (x, y) in enumerate(zip(self.agents_pos, self.goals_pos)):
            if np.array_equal(x, y):
                assert i <= len(rewards) - 1, write_file('rewards length too short')
                before_reward = rewards[i]
                rewards[i] = self.reward_fn['finish'] / len(self.agents_pos)
                content = 'agent {} has {} reward, before reward = {}'.format(i, rewards[i], before_reward)
                write_file(content)

        # check done
        if np.array_equal(self.agents_pos, self.goals_pos):
            done = True
            rewards = [self.reward_fn['finish'] for _ in range(self.num_agents)]
            content = 'done is True, four agents is at goal'
            write_file(content, name='agent_finish.txt')
        else:
            done = False

        info = {'step': self.steps - 1}

        # make sure no overlapping agents
        if np.unique(self.agents_pos, axis=0).shape[0] < self.num_agents:
            print(self.steps)
            print(self.map)
            print(self.agents_pos)
            raise RuntimeError('unique')

        # update last actions
        self.last_actions = np.zeros((self.num_agents, 5, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1),
                                     dtype=np.bool)
        self.last_actions[np.arange(self.num_agents), np.array(actions)] = 1

        return self.observe(), rewards, done, info

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
        # 合并动态地图
        dynamic_ped_map = 0
        if isinstance(self.dyn_map, pd.DataFrame) and self.dyn_map:  # 数据类型判断
            dynamic_ped_map = self.dyn_map
        else:
            # 获取行人x y 坐标
            coorlist = self.dynamic_ped.pde_df.iloc[self.t].tolist()
            # 得到栅格化中的位置
            dynamic_ped_map = self.dynamic_ped.get_pedcoor(coorlist)
            self.t = self.t + 1  # 时间加一
            # todo 只是测试，之后删掉
            if self.t >= len(self.dynamic_ped.pde_df):
                self.t = 0
        # 静态地图和动态地图合并
        # self.map = self.static_obs.static_map
        self.map = self.static_obs.static_map + dynamic_ped_map
        # 将大于1的全部置位1
        self.map[np.where(self.map >= 1)] = 1

        obs = np.zeros((self.num_agents, 6, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1), dtype=np.bool)

        # 0 represents obstacle to match 0 padding in CNN 地图上下左右都进行填充
        obstacle_map = np.pad(self.map, self.obs_radius, 'constant', constant_values=0)  # 对边缘进行填充

        agent_map = np.zeros((self.map_size), dtype=np.bool)
        agent_map[self.agents_pos[:, 0], self.agents_pos[:, 1]] = 1
        agent_map = np.pad(agent_map, self.obs_radius, 'constant', constant_values=0)

        for i, agent_pos in enumerate(self.agents_pos):
            x, y = agent_pos

            obs[i, 0] = agent_map[x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]
            obs[i, 0, self.obs_radius, self.obs_radius] = 0
            obs[i, 1] = obstacle_map[x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]
            obs[i, 2:] = self.heuri_map[i, :, x:x + 2 * self.obs_radius + 1, y:y + 2 * self.obs_radius + 1]

        # obs = np.concatenate((obs, self.last_actions), axis=1)
        return obs, np.copy(self.agents_pos)

    def render(self):
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
        # plt.xlabel('step: {}'.format(self.steps))

        # add text in plot
        self.imgs.append([])
        if hasattr(self, 'texts'):
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(self.agents_pos, self.goals_pos)):
                self.texts[i].set_position((agent_y, agent_x))
                self.texts[i].set_text(i)
        else:
            self.texts = []
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(self.agents_pos, self.goals_pos)):
                text = plt.text(agent_y, agent_x, i, color='black', ha='center', va='center')
                plt.text(goal_y, goal_x, i, color='black', ha='center', va='center')
                self.texts.append(text)

        plt.imshow(color_map[map], animated=True)

        plt.show()
        # plt.ion()
        plt.pause(0.5)

    def close(self, save=False):
        plt.close()
        del self.fig
