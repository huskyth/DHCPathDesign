from socket import *
import pandas as pd
from itertools import chain
import random
import numpy as np
import os
from dyn_environment import Environment
from model import Network
import configs
from construct_map.transform2coor import transform2coor
from utils.model_save_load_tool import *

torch.manual_seed(configs.test_seed)
np.random.seed(configs.test_seed)
random.seed(configs.test_seed)
test_num = 200
device = 'cpu'
torch.set_num_threads(1)

tcp_server = socket(AF_INET, SOCK_STREAM)

# netstat -tunlp | grep 1116
address = ('10.101.104.49', 1116)
tcp_server.bind(address)
tcp_server.listen(128)
print('-------------- Start listening to port 1116 -----------------')
client, adr = tcp_server.accept()


def send(data):
    client.send(data.encode('utf-8'))


pde_df = pd.DataFrame(columns=['p13', 'p1', 'p4', 'p7', 'p6', 'p11', 'p12',
                               'p5', 'p8', 'p14', 'p9', 'p10', 'p2', 'p3'])
model_name = TEST_MODEL_NAME
print('----------test model in unity {}----------'.format(model_name))
network = Network()
network.eval()
network.to(device)
weight_file = os.path.join(configs.save_path, model_name)
state_dict = model_load(weight_file, device)['model_state']
network.load_state_dict(state_dict)
network.eval()
network.share_memory()
env = Environment()
# Todo:将起始点传给unity
start, goal = transform2coor(env.agents_pos, env.goals_pos)
print("start pos:", list(start), "\ngoal pos:", list(goal))
send(str(list(chain.from_iterable(start))))
obs, pos = env.observe()
done = False
network.reset()
actions = []

while True:
    pedestrian_data = client.recv(1024).decode('utf-8')
    print("动态行人数据：", pedestrian_data)
    pde_df.loc[len(pde_df)] = eval(pedestrian_data)
    step = 0
    print("max episode length:", configs.max_episode_length)
    if not done and env.steps < configs.max_episode_length:
        # todo:传回action
        actions, q_val, hidden, comm_mask = network.step(torch.as_tensor(obs.astype(np.float32)),
                                                         torch.as_tensor(pos.astype(np.float32)))
        actions_Str = ''.join(str(e) for e in actions)
        send(actions_Str)

        (obs, pos), rew, done, info = env.step(actions, pde_df)
        step += 1
    else:
        ret = np.array_equal(env.agents_pos, env.goals_pos), step
        print(ret)

client_socket.close()
client.close()
