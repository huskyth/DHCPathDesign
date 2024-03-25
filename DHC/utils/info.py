import os
import threading


def print_process_info(id):
    print('actor id = ', id)
    print("当前进程：", os.getpid(), " 父进程：", os.getppid())
    t = threading.currentThread()
    print('Thread id : %d' % t.ident)
    print('actor id = ', id)

#
# import torch
#
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
