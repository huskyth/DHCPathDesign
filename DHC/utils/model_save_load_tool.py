import torch

TEST_MODEL_NAME = 'saved_model.pth'


def model_save(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def model_load(path):
    import os
    path = os.path.join(r"C:\Users\qq162\Desktop\DHCPathDesign\DHC\models", TEST_MODEL_NAME)
    ckt = torch.load(path)
    return ckt['model_state_dict'], ckt['optimizer_state_dict']
