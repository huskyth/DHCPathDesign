from torch.utils.tensorboard import SummaryWriter


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
        if x % 2 == 0:
            agents_pos[x][0], agents_pos[x][1] = x, x + 1
            goals_pos[x][0], goals_pos[x][1] = x, x + 1
        else:
            agents_pos[x][0], agents_pos[x][1] = x, x + 1
            goals_pos[x][0], goals_pos[x][1] = x, x + 1
    print(agents_pos)
    print(goals_pos)
    print(np.array_equal(agents_pos, goals_pos))
