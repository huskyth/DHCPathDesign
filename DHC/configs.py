DEBUG_MODE = False

communication = False
############################################################
####################    environment     ####################
############################################################
map_length = 50
num_agents = 1
obs_radius = 0
reward_fn = dict(move=-0.5,
                 stay_on_goal=1,
                 stay_off_goal=-0.5,
                 collision=-1,
                 finish=3)
from pathlib import Path
import os

FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parent
MODEL_FILE = PROJECT_ROOT / 'models'
if not MODEL_FILE.exists():
    os.mkdir(str(MODEL_FILE))
obs_shape = (6, 2 * obs_radius + 1, 2 * obs_radius + 1)
action_dim = 5

############################################################
####################         DQN        ####################
############################################################

num_actors = 8 if not DEBUG_MODE else 1
log_interval = 10
training_times = 60000
save_interval = 2000
gamma = 0.99
batch_size = 1024 if not DEBUG_MODE else 2
learning_starts = 50000
target_network_update_freq = 2000
save_path = str(MODEL_FILE)
max_episode_length = 100 if not DEBUG_MODE else 2
seq_len = 1
load_model = None

actor_update_steps = 1500
actor_random_generate_acceleration = 4000

# gradient norm clipping
grad_norm_dqn = 40

# n-step forward
forward_steps = 1

# global buffer
episode_capacity = 2048

# prioritized replay
prioritized_replay_alpha = 0.6
prioritized_replay_beta = 0.4

# curriculum learning
init_env_settings = (num_agents, 10)
map_size = (56, 22)
max_num_agents = num_agents
max_map_lenght = 40
pass_rate = 0.9

# dqn network setting
cnn_channel = 128
hidden_dim = 6

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
