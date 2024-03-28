import torch

TEST_MODEL_NAME = '2022-10-26-18-12000-0.86.pth'


def model_save(model, path):
    torch.save(model, path)


def model_load(path, device):
    return torch.load(path, map_location=device)
