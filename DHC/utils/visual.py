import random

from torch.utils.tensorboard import SummaryWriter


# C:\Users\Administrator>ping 10.101.104.49
#
# 正在 Ping 10.101.104.49 具有 32 字节的数据:
# 来自 10.101.104.49 的回复: 字节=32 时间<1ms TTL=60
# 来自 10.101.104.49 的回复: 字节=32 时间<1ms TTL=60
# 来自 10.101.104.49 的回复: 字节=32 时间<1ms TTL=60
# 来自 10.101.104.49 的回复: 字节=32 时间<1ms TTL=60
#
# 10.101.104.49 的 Ping 统计信息:
#     数据包: 已发送 = 4，已接收 = 4，丢失 = 0 (0% 丢失)，
# 往返行程的估计时间(以毫秒为单位):
#     最短 = 0ms，最长 = 0ms，平均 = 0ms

# tensorboard --logdir=reward_visual --host=10.101.104.49


def init_summary_writer(dir_name='visual'):
    writer = SummaryWriter(dir_name)
    return writer


def plot(writer, x, y, title):
    writer.add_scalar(title, y, x)


if __name__ == '__main__':
    print('his')
    num_agents = 4
    import numpy as np

    agents_pos = np.empty((num_agents, 2), dtype=np.int)
    goals_pos = np.empty((num_agents, 2), dtype=np.int)
    for x in range(num_agents):
        if x % 2 ==0:
            agents_pos[x][0], agents_pos[x][1] = x,x+1
            goals_pos[x][0], goals_pos[x][1] = x,x+1
        else:
            agents_pos[x][0], agents_pos[x][1] = x,x+1
            goals_pos[x][0], goals_pos[x][1] = x,x+1
    print(agents_pos)
    print(goals_pos)
    print(np.array_equal(agents_pos,goals_pos))

    # for i, (x, y) in enumerate(zip(agents_pos, goals_pos)):
    #     print('x = {},y = {}'.format(x, y))
    #     if np.array_equal(x, y):
    #         print('agent {} has reward'.format(i))
    #     else:
    #         print('agent {} has no reward'.format(i))
