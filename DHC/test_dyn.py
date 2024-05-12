import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dyn_environment import Environment
from model import Network
import configs
import time
import os
from utils.model_save_load_tool import *

test_num = 200
device = 'cpu'
torch.set_num_threads(1)


def test_model(model_name):
    '''
    test model in 'models' file with model number
    '''
    weight_file = os.path.join(configs.save_path, model_name)
    state_dict = model_load(weight_file, device)
    network = state_dict
    network.eval()
    network.share_memory()

    print('----------test model {}----------'.format(model_name))


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

        (obs, pos), rew, done, info = env.step(actions)
        step += 1
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

    weight_file = os.path.join(configs.save_path, model_name)
    network = model_load(weight_file, device)

    env = Environment()

    fig = plt.figure()

    done = False
    obs, pos, _ = env.observe()

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

        actions, _, _ = network.step(torch.from_numpy(obs.astype(np.float32)).to(device))
        (obs, pos, _), _, done, _ = env.step(actions)

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
    test_model(TEST_MODEL_NAME)
    make_animation(model_name=TEST_MODEL_NAME)
    network = Network()
