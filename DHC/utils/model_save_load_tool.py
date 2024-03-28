import torch

RESUME_MODEL_NAME = ''
TEST_MODEL_NAME = '2022-10-26-18-12000-0.86.pth'


def model_save(model, path, counter, avg_loss, avg_reward, avg_finish_cases, avg_step):
    state = {'model_state': model.state_dict(), 'counter': counter, 'avg_loss': avg_loss,
             'avg_reward': avg_reward, 'avg_finish_cases': avg_finish_cases, 'avg_step': avg_step}
    torch.save(state, path)


def model_load(path, device):
    return torch.load(path, map_location=device)
