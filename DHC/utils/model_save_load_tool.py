import torch

TEST_MODEL_NAME = '2025-04-12-13-151985.pth'


def model_save(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def model_load(path):
    ckt = torch.load(path)
    return ckt['model_state_dict'], ckt['optimizer_state_dict']
