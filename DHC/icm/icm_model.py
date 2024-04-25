import torch.nn as nn
import torch.nn.functional as F
import torch

from DHC import configs
from DHC.configs import cnn_channel, num_agents, hidden_dim, batch_size
from DHC.model import ResBlock, CommBlock
from torch.cuda.amp import autocast


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

        self.pred_module1 = nn.Linear(hidden_dim + num_actions, 256)
        self.pred_module2 = nn.Linear(256, hidden_dim)

        self.invpred_module1 = nn.Linear(hidden_dim * 2, 256)
        self.invpred_module2 = nn.Linear(256, num_actions)
        self.input_shape = configs.obs_shape
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(self.input_shape[0], cnn_channel, 3, 1),
            nn.ReLU(True),
            ResBlock(cnn_channel),
            ResBlock(cnn_channel),
            ResBlock(cnn_channel),
            nn.Conv2d(cnn_channel, 16, 1, 1),
            nn.ReLU(True),
            nn.Flatten(),
        )
        self.latent_dim = 16 * 7 * 7
        self.recurrent = nn.GRUCell(self.latent_dim, hidden_dim)
        self.comm = CommBlock(hidden_dim)

    def get_feature(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.contiguous().view(x.size(0), -1)))
        return x

    def forward(self, x):
        # get feature
        feature_x = self.get_feature(x)
        return feature_x

    def encode(self, x, comm_mask, steps, hidden):
        max_steps = x.size(1)
        # get feature
        x = x.transpose(1, 2)

        x = x.contiguous().view(-1, *self.input_shape)
        latent_x = self.obs_encoder(x)

        latent = latent_x.view(configs.batch_size * num_agents, max_steps, self.latent_dim).transpose(0, 1)

        hidden_buffer = []
        for i in range(max_steps):
            # hidden size: batch_size*num_agents x self.hidden_dim
            hidden = self.recurrent(latent[i], hidden)
            hidden = hidden.view(batch_size, num_agents, hidden_dim)
            hidden = self.comm(hidden, comm_mask[:, i])
            # only hidden from agent 0
            hidden_buffer.append(hidden[:, 0])
            hidden = hidden.view(batch_size * num_agents, hidden_dim)

        # hidden buffer size: batch_size x seq_len x self.hidden_dim
        hidden_buffer = torch.stack(hidden_buffer).transpose(0, 1)
        # hidden size: batch_size x self.hidden_dim
        hidden = hidden_buffer[torch.arange(configs.batch_size), steps - 1]
        return hidden

    @autocast()
    def get_full(self, x, x_next, a_vec, comm_mask, steps, hidden):
        feature_x = self.encode(x, comm_mask, steps, hidden)

        feature_x_next = self.encode(x_next, comm_mask, steps, hidden)

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
