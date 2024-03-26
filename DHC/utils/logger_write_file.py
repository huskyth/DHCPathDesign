import configs
import os

LOG_FILE_NAME = 'log_file.txt'


def write_file(content, name=LOG_FILE_NAME, swich=False):
    if not swich:
        return
    with open(configs.save_path + os.sep + name, 'a') as f:
        f.write(content + '\n')


if __name__ == '__main__':
    pass
