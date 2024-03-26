communication = False
# 10.101.104.49
############################################################
####################    environment     ####################
############################################################
map_length = 50
# num_agents = 2
num_agents = 4
obs_radius = 4
# obs_radius = 1
reward_fn = dict(move=-0.075,
                 stay_on_goal=0,
                 stay_off_goal=-0.075,
                 collision=-0.5,
                 finish=3)
from pathlib import Path
import os

FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parent.parent.parent.parent
MODEL_FILE = PROJECT_ROOT / 'models'
if not MODEL_FILE.exists():
    os.mkdir(str(MODEL_FILE))
print(MODEL_FILE)
obs_shape = (6, 2 * obs_radius + 1, 2 * obs_radius + 1)
action_dim = 5

############################################################
####################         DQN        ####################
############################################################

# basic training setting
# num_actors = 16
num_actors = 16
log_interval = 10
training_times = 60000
# training_times = 600
save_interval = 2000
# save_interval = 500
gamma = 0.99
batch_size = 512
learning_starts = 50000
# learning_starts = 50
target_network_update_freq = 1000
# target_network_update_freq = 20
save_path = str(MODEL_FILE)
# max_episode_length = 256
# max_episode_length = 1024 #todo:需要修改
max_episode_length = 512
seq_len = 16
load_model = None

max_episode_length = max_episode_length

actor_update_steps = 400

# gradient norm clipping
grad_norm_dqn = 40

# n-step forward
forward_steps = 2

# global buffer
episode_capacity = 2048

# prioritized replay
prioritized_replay_alpha = 0.6
prioritized_replay_beta = 0.4

# curriculum learning
# init_env_settings = (1, 10)
init_env_settings = (4, 10)  # 一维修改agent大小
map_size = (56, 22)  # unity地图大小
# max_num_agents = 2
max_num_agents = 4
max_map_lenght = 40
pass_rate = 0.9

# dqn network setting
cnn_channel = 128
hidden_dim = 256

# communication
max_comm_agents = 3  # including agent itself, means one can at most communicate with (max_comm_agents-1) agents

# communication block
num_comm_layers = 2
num_comm_heads = 2

test_seed = 2022
num_test_cases = 200
# map length, number of agents, density
test_env_settings = ((40, 4, 0.3), (40, 8, 0.3), (40, 16, 0.3), (40, 32, 0.3), (40, 64, 0.3),
                     (80, 4, 0.3), (80, 8, 0.3), (80, 16, 0.3), (80, 32, 0.3), (80, 64, 0.3))
