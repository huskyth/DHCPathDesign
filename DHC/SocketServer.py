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
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)

# 创建套接字
tcp_server = socket(AF_INET, SOCK_STREAM)
# 绑定ip，port
# 这里ip默认本机
# netstat -tunlp | grep 1116
address = ('10.101.104.49', 1116)
tcp_server.bind(address)
# 启动被动连接
# 多少个客户端可以连接
tcp_server.listen(128)
print('-------------- Start listening to port 1116 -----------------')
client, adr = tcp_server.accept()


def send(data):
    # data=1
    client.send(data.encode('utf-8'))  # 发送


# 新建动态行人矩阵
pde_df = pd.DataFrame(columns=['p13', 'p1', 'p4', 'p7', 'p6', 'p11', 'p12',
                               'p5', 'p8', 'p14', 'p9', 'p10', 'p2', 'p3'])
model_name = TEST_MODEL_NAME
print('----------test model in unity {}----------'.format(model_name))
network = Network()
network.eval()
network.to(device)
# 加载模型
weight_file = os.path.join(configs.save_path, model_name)
state_dict = model_load(weight_file, device)['model_state']
network.load_state_dict(state_dict)
network.eval()
network.share_memory()  # 允许数据处于一种特殊的状态，可以在不需要拷贝的情况下，任何进程都可以直接使用该数据
env = Environment()  # 创建环境
# todo:将起始点传给unity
start, goal = transform2coor(env.agents_pos, env.goals_pos)
# action，env.agents_pos，env.goals_pos需要传给unity
print("start pos:", list(start), "\ngoal pos:", list(goal))
send(str(list(chain.from_iterable(start))))
obs, pos = env.observe()
done = False
network.reset()
actions = []

while True:
    # todo：行人预测
    # while len(pde_df)<=8,
    # unity发过来行人数据
    pedestrian_data = client.recv(1024).decode('utf-8')  # 接收1024给字节,这里recv接收的不再是元组，区别UDP
    # from_client_msg = from_client_msg.decode("cp936")
    print("动态行人数据：", pedestrian_data)
    # 构建行人动态坐标矩阵
    pde_df.loc[len(pde_df)] = eval(pedestrian_data)
    # 进行一次交互
    step = 0
    print("max episode length:", configs.max_episode_length)
    if not done and env.steps < configs.max_episode_length:
        # todo:传回action
        actions, q_val, hidden, comm_mask = network.step(torch.as_tensor(obs.astype(np.float32)),
                                                         torch.as_tensor(pos.astype(np.float32)))
        # 传回actions
        actions_Str = ''.join(str(e) for e in actions)
        send(actions_Str)

        (obs, pos), rew, done, info = env.step(actions, pde_df)  # 传入行人数据
        step += 1
    else:
        ret = np.array_equal(env.agents_pos, env.goals_pos), step
        print(ret)

# 关闭套接字
# 关闭为这个客户端服务的套接字，就意味着为不能再为这个客户端服务了
# 如果还需要服务，只能再次重新连
client_socket.close()
client.close()  # 结束后关闭
