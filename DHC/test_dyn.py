'''create test set and test model'''
'''create test set and test model'''
import random
import pickle
import multiprocessing as mp
from typing import Union
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dyn_environment import Environment
from model import Network
import configs
from construct_map.transform2coor import transform2coor
import time
import os
from utils.model_save_load_tool import *

# torch.manual_seed(configs.test_seed)
# np.random.seed(configs.test_seed)
# random.seed(configs.test_seed)
test_num = 200
device = 'cpu'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)


# 测试用例，要用到
def test_model(model_name):
    '''
    test model in 'models' file with model number
    '''
    network = Network()
    network.eval()
    network.to(device)
    weight_file = os.path.join(configs.save_path, model_name)
    state_dict = model_load(weight_file, device)
    network.load_state_dict(state_dict['model_state'])
    network.eval()
    network.share_memory()  # 允许数据处于一种特殊的状态，可以在不需要拷贝的情况下，任何进程都可以直接使用该数据

    print('----------test model {}----------'.format(model_name))

    # 进行测试
    is_finish, steps = test_one_case(network)
    # print(ret)

    # success = 0
    # avg_step = 0
    # for i, j in ret:
    #     success += i
    #     avg_step += j
    #
    # print("success rate: {:.2f}%".format(success / len(ret) * 100))
    # print("average step: {}".format(avg_step / len(ret)))


def test_one_case(network):
    env = Environment()
    obs, pos = env.observe()

    done = False
    network.reset()

    step = 0
    print("max episode length:", configs.max_episode_length)
    while not done and env.steps < configs.max_episode_length:
        # todo:传回action
        actions, q_val, hidden, comm_mask = network.step(torch.as_tensor(obs.astype(np.float32)),
                                                         torch.as_tensor(pos.astype(np.float32)))
        # 转换到unity坐标系
        start, goal = transform2coor(env.agents_pos, env.goals_pos)
        # action，env.agents_pos，env.goals_pos需要传给unity
        # print("actions:", actions, "\nstart pos:", list(start), "\ngoal pos:", list(goal))

        (obs, pos), rew, done, info = env.step(actions)  # reward没有用上
        step += 1
    # 是否达到终点、走了多少步
    return np.array_equal(env.agents_pos, env.goals_pos), step


def make_animation(model_name, steps: int = 1000):
    '''
    visualize running results
    model_name: model number in 'models' file
    steps: how many steps to visualize in test case
    '''
    color_map = np.array([[255, 255, 255],  # white
                          [190, 190, 190],  # gray
                          [0, 191, 255],  # blue
                          [255, 165, 0],  # orange
                          [0, 250, 154]])  # green

    network = Network()
    network.eval()
    network.to(device)
    weight_file = os.path.join(configs.save_path, model_name)
    state_dict = model_load(weight_file, device)
    network.load_state_dict(state_dict['model_state'])

    env = Environment()

    fig = plt.figure()

    done = False
    obs, pos = env.observe()

    imgs = []
    while not done and env.steps < steps:
        imgs.append([])
        map = np.copy(env.map)
        for agent_id in range(env.num_agents):
            if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                map[tuple(env.agents_pos[agent_id])] = 4
            else:
                map[tuple(env.agents_pos[agent_id])] = 2
                map[tuple(env.goals_pos[agent_id])] = 3
        map = map.astype(np.uint8)

        img = plt.imshow(color_map[map], animated=True)

        imgs[-1].append(img)

        for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(env.agents_pos, env.goals_pos)):
            text = plt.text(agent_y, agent_x, i, color='black', ha='center', va='center')
            imgs[-1].append(text)
            text = plt.text(goal_y, goal_x, i, color='black', ha='center', va='center')
            imgs[-1].append(text)

        actions, _, _, _ = network.step(torch.from_numpy(obs.astype(np.float32)).to(device),
                                        torch.from_numpy(pos.astype(np.float32)).to(device))
        (obs, pos), _, done, _ = env.step(actions)

    if done and env.steps < steps:
        map = np.copy(env.map)
        for agent_id in range(env.num_agents):
            if np.array_equal(env.agents_pos[agent_id], env.goals_pos[agent_id]):
                map[tuple(env.agents_pos[agent_id])] = 4
            else:
                map[tuple(env.agents_pos[agent_id])] = 2
                map[tuple(env.goals_pos[agent_id])] = 3
        map = map.astype(np.uint8)

        img = plt.imshow(color_map[map], animated=True)
        for _ in range(steps - env.steps):
            imgs.append([])
            imgs[-1].append(img)
            for i, ((agent_x, agent_y), (goal_x, goal_y)) in enumerate(zip(env.agents_pos, env.goals_pos)):
                text = plt.text(agent_y, agent_x, i, color='black', ha='center', va='center')
                imgs[-1].append(text)
                text = plt.text(goal_y, goal_x, i, color='black', ha='center', va='center')
                imgs[-1].append(text)

    ani = animation.ArtistAnimation(fig, imgs, interval=600, blit=True, repeat_delay=1000)

    # ani.save('videos/{}_{}_{}_{}.mp4'.format(model_name, *test_set_name))
    ticks = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    ani.save('videos/{}.gif'.format(ticks))


def create_test(test_env_settings, num_test_cases):
    for map_length, num_agents, density in test_env_settings:

        name = './test_set/{}length_{}agents_{}density.pth'.format(map_length, num_agents, density)
        print('-----{}length {}agents {}density-----'.format(map_length, num_agents, density))

        tests = []

        env = Environment(fix_density=density, num_agents=num_agents, map_length=map_length)

        for _ in tqdm(range(num_test_cases)):
            tests.append((np.copy(env.map), np.copy(env.agents_pos), np.copy(env.goals_pos)))
            env.reset(num_agents=num_agents, map_length=map_length)
        print()
        with open(name, 'wb') as f:
            pickle.dump(tests, f)


def test_while_training(network, num=100):
    print('start test')
    network.eval()
    network.to('cpu')
    all_is_finish = 0
    all_steps = 0
    for x in range(num):
        is_finish, steps = test_one_case(network)
        all_is_finish += 1 if is_finish else 0
        all_steps += steps
    network.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    return all_is_finish / num, all_steps / num


if __name__ == '__main__':
    # create_test(test_env_settings=configs.test_env_settings, num_test_cases=configs.num_test_cases)
    test_model(TEST_MODEL_NAME)
    make_animation(model_name=TEST_MODEL_NAME)
    # test_while_training()
    network = Network()
    # weight_file = os.path.join(configs.save_path, TEST_MODEL_NAME)
    # state_dict = model_load(weight_file, device)
    # print('test model {}'.format(TEST_MODEL_NAME))
    # network.load_state_dict(state_dict['model_state'])
    # avg_finish, avg_step = test_while_training(network, 1000)
    # print('avg_finish = {}, avg_step = {}'.format(avg_finish, avg_step))
