import sys

import gym, random, pickle, os.path, math, glob
from pathlib import Path

path = str(Path(__file__).parent.parent)
if path not in sys.path:
    sys.path.append(path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import pdb

from DHC.icm.icm_model import ICM
from atari_wrappers import make_atari, wrap_deepmind, LazyFrames
from IPython.display import clear_output
from tensorboardX import SummaryWriter

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args,
                                                                                                                **kwargs)
# Create and wrap the environment
env = make_atari('PongNoFrameskip-v4')  # only use in no frameskip environment
env = wrap_deepmind(env, scale=False, frame_stack=True)
n_actions = env.action_space.n
state_dim = env.observation_space.shape

# env.render()
test = env.reset()
for i in range(100):
    test = env.step(env.action_space.sample())[0]

plt.imshow(test._force()[..., 0])


# plt.imshow(env.render("rgb_array"))
# env.close()


class DQN(nn.Module):
    def __init__(self, in_channels=4, num_actions=5):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.contiguous().view(x.size(0), -1)))
        return self.fc5(x)


class ICM(nn.Module):
    def __init__(self, in_channels=4, num_actions=5):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(ICM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)

        self.pred_module1 = nn.Linear(512 + num_actions, 256)
        self.pred_module2 = nn.Linear(256, 512)

        self.invpred_module1 = nn.Linear(512 + 512, 256)
        self.invpred_module2 = nn.Linear(256, num_actions)

    def get_feature(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return x

    def forward(self, x):
        # get feature
        feature_x = self.get_feature(x)
        return feature_x

    def get_full(self, x, x_next, a_vec):
        # get feature
        feature_x = self.get_feature(x)
        feature_x_next = self.get_feature(x_next)

        pred_s_next = self.pred(feature_x, a_vec)  # predict next state feature
        pred_a_vec = self.invpred(feature_x, feature_x_next)  # (inverse) predict action

        return pred_s_next, pred_a_vec, feature_x_next

    def pred(self, feature_x, a_vec):
        # Forward prediction: predict next state feature, given current state feature and action (one-hot)
        pred_s_next = F.relu(self.pred_module1(torch.cat([feature_x, a_vec.float()], dim=-1).detach()))
        pred_s_next = self.pred_module2(pred_s_next)
        return pred_s_next

    def invpred(self, feature_x, feature_x_next):
        # Inverse prediction: predict action (one-hot), given current and next state features
        pred_a_vec = F.relu(self.invpred_module1(torch.cat([feature_x, feature_x_next], dim=-1)))
        pred_a_vec = self.invpred_module2(pred_a_vec)
        return F.softmax(pred_a_vec, dim=-1)


class Memory_Buffer(object):
    def __init__(self, memory_size=1000):
        self.buffer = []
        self.memory_size = memory_size
        self.next_idx = 0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) <= self.memory_size:  # buffer not full
            self.buffer.append(data)
        else:  # buffer is full
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.size() - 1)
            data = self.buffer[idx]
            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones

    def size(self):
        return len(self.buffer)


class ICM_DQNAgent:
    def __init__(self, in_channels=1, action_space=[], USE_CUDA=False, memory_size=10000, epsilon=1, lr=1e-4,
                 forward_scale=0.8, inverse_scale=0.2, Qloss_scale=0.1, intrinsic_scale=1, use_extrinsic=True):
        self.epsilon = epsilon
        self.action_space = action_space
        # param for ICM
        self.forward_scale = forward_scale  # scale for loss function of forward prediction model, 0.8
        self.inverse_scale = inverse_scale  # scale for loss function of inverse prediction model, 0.2
        self.Qloss_scale = Qloss_scale  # scale for loss function of Q value, 1
        self.intrinsic_scale = intrinsic_scale  # scale for intrinsic reward, 1
        self.use_extrinsic = use_extrinsic  # whether use extrinsic rewards, if False, only intrinsic reward generated from ICM is used

        self.memory_buffer = Memory_Buffer(memory_size)
        self.DQN = DQN(in_channels=in_channels, num_actions=action_space.n)
        self.DQN_target = DQN(in_channels=in_channels, num_actions=action_space.n)
        self.DQN_target.load_state_dict(self.DQN.state_dict())
        self.ICM = ICM(in_channels=in_channels, num_actions=action_space.n)

        self.USE_CUDA = USE_CUDA
        if USE_CUDA:
            self.DQN = self.DQN.cuda()
            self.DQN_target = self.DQN_target.cuda()
            self.ICM = self.ICM.cuda()
        self.optimizer = optim.Adam(list(self.DQN.parameters()) + list(self.ICM.parameters()), lr=lr)

    def observe(self, lazyframe):
        # from Lazy frame to tensor
        state = torch.from_numpy(lazyframe._force().transpose(2, 0, 1)[None] / 255).float()
        if self.USE_CUDA:
            state = state.cuda()
        return state

    def value(self, state):
        q_values = self.DQN(state)
        return q_values

    def act(self, state, epsilon=None):
        """
        sample actions with epsilon-greedy policy
        recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
        """
        if epsilon is None: epsilon = self.epsilon

        q_values = self.value(state).cpu().detach().numpy()
        if random.random() < epsilon:
            aciton = random.randrange(self.action_space.n)
        else:
            aciton = q_values.argmax(1)[0]
        return aciton

    def compute_td_loss(self, states, actions, rewards, next_states, is_done, gamma=0.99):
        """ Compute td loss using torch operations only. Use the formula above. """
        actions = torch.tensor(actions).long()  # shape: [batch_size]
        rewards = torch.tensor(rewards, dtype=torch.float)  # shape: [batch_size]
        is_done = torch.tensor(is_done).type(torch.bool)  # shape: [batch_size]

        if self.USE_CUDA:
            actions = actions.cuda()
            rewards = rewards.cuda()
            is_done = is_done.cuda()

        # get q-values for all actions in current states
        predicted_qvalues = self.DQN(states)

        # get ICM results
        a_vec = F.one_hot(actions, num_classes=self.action_space.n)  # convert action from int to one-hot format
        pred_s_next, pred_a_vec, feature_x_next = self.ICM.get_full(states, next_states, a_vec)
        # calculate forward prediction and inverse prediction loss
        forward_loss = F.mse_loss(pred_s_next, feature_x_next.detach(), reduction='none')
        inverse_pred_loss = F.cross_entropy(pred_a_vec, actions.detach(), reduction='none')

        # calculate rewards
        intrinsic_rewards = self.intrinsic_scale * forward_loss.mean(-1)
        total_rewards = intrinsic_rewards.clone()
        if self.use_extrinsic:
            total_rewards += rewards

        # select q-values for chosen actions
        predicted_qvalues_for_actions = predicted_qvalues[
            range(states.shape[0]), actions
        ]

        # compute q-values for all actions in next states
        predicted_next_qvalues = self.DQN_target(next_states)  # YOUR CODE

        # compute V*(next_states) using predicted next q-values
        next_state_values = predicted_next_qvalues.max(-1)[0]  # YOUR CODE

        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        target_qvalues_for_actions = total_rewards + gamma * next_state_values  # YOUR CODE

        # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        target_qvalues_for_actions = torch.where(
            is_done, total_rewards, target_qvalues_for_actions)

        # mean squared error loss to minimize
        # loss = torch.mean((predicted_qvalues_for_actions -
        #                   target_qvalues_for_actions.detach()) ** 2)
        Q_loss = F.smooth_l1_loss(predicted_qvalues_for_actions, target_qvalues_for_actions.detach())
        loss = self.Qloss_scale * Q_loss + self.forward_scale * forward_loss.mean() + self.inverse_scale * inverse_pred_loss.mean()

        return loss, Q_loss.item(), forward_loss.mean().item(), inverse_pred_loss.mean().item(), intrinsic_rewards.mean().item()

    def sample_from_buffer(self, batch_size):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(batch_size):
            idx = random.randint(0, self.memory_buffer.size() - 1)
            data = self.memory_buffer.buffer[idx]
            frame, action, reward, next_frame, done = data
            states.append(self.observe(frame))
            actions.append(action)
            rewards.append(reward)
            next_states.append(self.observe(next_frame))
            dones.append(done)
        return torch.cat(states), actions, rewards, torch.cat(next_states), dones

    def learn_from_experience(self, batch_size):
        if self.memory_buffer.size() > batch_size:
            states, actions, rewards, next_states, dones = self.sample_from_buffer(batch_size)
            td_loss, Q_loss, forward_loss, inverse_pred_loss, intrinsic_rewards = self.compute_td_loss(states, actions,
                                                                                                       rewards,
                                                                                                       next_states,
                                                                                                       dones)
            self.optimizer.zero_grad()
            td_loss.backward()
            for param in list(self.DQN.parameters()) + list(self.ICM.parameters()):
                param.grad.data.clamp_(-1, 1)

            self.optimizer.step()
            return (td_loss.item(), Q_loss, forward_loss, inverse_pred_loss, intrinsic_rewards)
        else:
            return (0, 0, 0, 0, 0)


# if __name__ == '__main__':

# Training DQN in PongNoFrameskip-v4
env = make_atari('PongNoFrameskip-v4')
env = wrap_deepmind(env, scale=False, frame_stack=True)

gamma = 0.99
epsilon_max = 1
epsilon_min = 0.01
eps_decay = 50000
frames = 1000000
USE_CUDA = True
learning_rate = 1e-4
max_buff = 100000
update_tar_interval = 1000
batch_size = 32
print_interval = 1000
log_interval = 1000
learning_start = 5000  # 10000
win_reward = 18  # Pong-v4
win_break = True

# param for ICM
forward_scale = 1  # scale for loss function of forward prediction model, 0.8
inverse_scale = 1  # scale for loss function of inverse prediction model, 0.2
Qloss_scale = 1  # scale for loss function of Q value, 1
intrinsic_scale = 100  # scale for intrinsic reward, 1
use_extrinsic = True  # whether use extrinsic rewards, if False, only intrinsic reward generated from ICM is used

action_space = env.action_space
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]
state_channel = env.observation_space.shape[2]
agent = ICM_DQNAgent(in_channels=state_channel, action_space=action_space, USE_CUDA=USE_CUDA, lr=learning_rate,
                     forward_scale=forward_scale, inverse_scale=inverse_scale, Qloss_scale=Qloss_scale,
                     intrinsic_scale=intrinsic_scale,
                     use_extrinsic=use_extrinsic)

frame = env.reset()

episode_reward = 0
all_rewards = []
losses = []
episode_num = 0
is_win = False
# tensorboard
summary_writer = SummaryWriter(log_dir="ICM_DQN3_Pong", comment="good_makeatari")

# e-greedy decay
epsilon_by_frame = lambda frame_idx: epsilon_min + (epsilon_max - epsilon_min) * math.exp(
    -1. * frame_idx / eps_decay)
# plt.plot([epsilon_by_frame(i) for i in range(10000)])
loss = 0
Q_loss_record = 0
forward_loss_record = 0
inverse_pred_loss_record = 0
intrinsic_rewards_rec = 0
for i in range(frames):
    epsilon = epsilon_by_frame(i)
    state_tensor = agent.observe(frame)
    action = agent.act(state_tensor, epsilon)

    next_frame, reward, done, _ = env.step(action)

    episode_reward += reward
    agent.memory_buffer.push(frame, action, reward, next_frame, done)
    frame = next_frame

    if agent.memory_buffer.size() >= learning_start:
        loss, Q_loss_record, forward_loss_record, inverse_pred_loss_record, intrinsic_rewards_rec = agent.learn_from_experience(
            batch_size)
        losses.append(loss)

    if i % print_interval == 0:
        print(
            "frames: %5d, reward: %5f, total_loss: %4f, forward_loss: %4f, inverse_pred_loss: %4f, Q_loss: %4f, intrinsic_rewards: %4f, epsilon: %5f, episode: %4d" %
            (i, np.mean(all_rewards[-10:]), loss, forward_loss_record, inverse_pred_loss_record, Q_loss_record,
             intrinsic_rewards_rec, epsilon, episode_num))
        summary_writer.add_scalar("Temporal Difference Loss", loss, i)
        summary_writer.add_scalar("Mean Reward", np.mean(all_rewards[-10:]), i)
        summary_writer.add_scalar("Epsilon", epsilon, i)

    if i % update_tar_interval == 0:
        agent.DQN_target.load_state_dict(agent.DQN.state_dict())

    if done:
        frame = env.reset()

        all_rewards.append(episode_reward)
        episode_reward = 0
        episode_num += 1
        avg_reward = float(np.mean(all_rewards[-100:]))

summary_writer.close()


def plot_training(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


plot_training(i, all_rewards, losses)
