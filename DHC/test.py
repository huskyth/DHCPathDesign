import random
import pickle
import multiprocessing as mp
from typing import Union
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from environment import Environment
from model import Network
import configs

torch.manual_seed(configs.test_seed)
np.random.seed(configs.test_seed)
random.seed(configs.test_seed)
test_num = 200
device = torch.device('cpu')
torch.set_num_threads(1)


def test_model(model_range: Union[int, tuple]):
    '''
    test model in 'models' file with model number 
    '''
    network = Network()
    network.eval()
    network.to(device)
    test_set = configs.test_env_settings
    pool = mp.Pool(mp.cpu_count())

    if isinstance(model_range, int):
        state_dict = torch.load('./models/{}.pth'.format(model_range), map_location=device)
        network.load_state_dict(state_dict)
        network.eval()
        network.share_memory()

        print('----------test model {}----------'.format(model_range))
        for case in test_set:
            print("test set: {} length {} agents {} density".format(case[0], case[1], case[2]))
            with open('./test_set/{}length_{}agents_{}density.pth'.format(case[0], case[1], case[2]), 'rb') as f:
                tests = pickle.load(f)  # 导入地图

            tests = [(test, network) for test in tests]
            ret = pool.map(test_one_case, tests)

            success = 0
            avg_step = 0
            for i, j in ret:
                success += i
                avg_step += j

            print("success rate: {:.2f}%".format(success / len(ret) * 100))
            print("average step: {}".format(avg_step / len(ret)))
            print()

    elif isinstance(model_range, tuple):
        for model_name in range(model_range[0], model_range[1] + 1, configs.save_interval):
            state_dict = torch.load('./models/{}.pth'.format(model_name), map_location=device)
            network.load_state_dict(state_dict)
            network.eval()
            network.share_memory()

            print('----------test model {}----------'.format(model_name))

            for case in test_set:
                print("test set: {} length {} agents {} density".format(case[0], case[1], case[2]))
                with open('./test_set/{}length_{}agents_{}density.pth'.format(case[0], case[1], case[2]), 'rb') as f:
                    tests = pickle.load(f)

                tests = [(test, network) for test in tests]
                ret = pool.map(test_one_case, tests)

                success = 0
                avg_step = 0
                for i, j in ret:
                    success += i
                    avg_step += j

                print("success rate: {:.2f}%".format(success / len(ret) * 100))
                print("average step: {}".format(avg_step / len(ret)))
                print()

            print('\n')


def test_one_case(args):
    env_set, network = args

    env = Environment()
    env.load(env_set[0], env_set[1], env_set[2])
    obs, pos = env.observe()

    done = False
    network.reset()

    step = 0
    while not done and env.steps < configs.max_episode_length:
        # todo:传回action
        actions, _, _, _ = network.step(torch.as_tensor(obs.astype(np.float32)),
                                        torch.as_tensor(pos.astype(np.float32)))
        (obs, pos), rew, done, _ = env.step(actions)  # reward没有用上
        step += 1
    return np.array_equal(env.agents_pos, env.goals_pos), step


def make_animation(model_name: int, test_set_name: tuple, test_case_idx: int, steps: int = 25):
    '''
    visualize running results
    model_name: model number in 'models' file
    test_set_name: (length, num_agents, density)
    test_case_idx: int, the test case index in test set
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
    state_dict = torch.load('models/{}.pth'.format(model_name), map_location=device)
    network.load_state_dict(state_dict)
    test_name = 'test_set/40length_16agents_0.3density.pth'
    with open(test_name, 'rb') as f:
        tests = pickle.load(f)

    env = Environment()
    env.load(tests[test_case_idx][0], tests[test_case_idx][1], tests[test_case_idx][2])

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

    ani.save('videos/{}_{}_{}_{}.gif'.format(model_name, *test_set_name))


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


if __name__ == '__main__':
    test_model(10000)
    make_animation(model_name=337500, test_set_name=tuple([40, 16, 0.3]), test_case_idx=0)
