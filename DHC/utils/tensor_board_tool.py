import os
import time
from torch.utils.tensorboard import SummaryWriter
import wandb
from pathlib import Path


def create_directory(path):
    if not os.path.exists(str(path)):
        os.mkdir(str(path))


ROOT_PATH = Path(__file__).parent.parent.parent

WANDB_PATH = ROOT_PATH / "wandb_log"
SUMMARY_PATH = ROOT_PATH / "summary_log"
create_directory(WANDB_PATH)
create_directory(SUMMARY_PATH)

log_dir = SUMMARY_PATH


class MySummary:

    def __init__(self, log_dir_name="default", use_wandb=True):
        log_path = str(log_dir / log_dir_name)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        self.writer = SummaryWriter(log_dir=log_path)
        self.use_wandb = use_wandb
        if use_wandb:
            ticks = str(time.time())
            wandb.login(key="613f55cae781fb261b18bad5ec25aa65766e6bc8")
            self.wandb_logger = wandb.init(project="WatermelonChess" + ticks, dir=WANDB_PATH)

    def add_float(self, x, y, title, x_name):
        self.writer.add_scalar(title, y, x)
        if self.use_wandb:
            self.wandb_logger.log({x_name: x, title: y})

    def close(self):
        self.writer.close()


if __name__ == '__main__':
    test = MySummary()
    for i in range(100):
        test.add_float(x=i, y=i ** 2, title="y=f(x)", x_name="x")
