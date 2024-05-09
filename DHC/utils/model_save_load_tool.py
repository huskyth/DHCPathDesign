import torch

TEST_MODEL_NAME = '2024-05-09-20-15999.pth'


def model_save(model, path):
    torch.save(model, path)


def model_load(path, device):
    return torch.load(path, map_location=device)
