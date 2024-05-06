import torch

TEST_MODEL_NAME = '2024-05-05-23-23998.pth'


def model_save(model, path):
    torch.save(model, path)


def model_load(path, device):
    return torch.load(path, map_location=device)
