import gc
import threading

from ray.thirdparty_files import psutil


def auto_garbage_collect(pct=80.0):
    while True:
        if psutil.virtual_memory().percent >= pct:
            print('Now is collecting garbage !!!!!!!!')
            gc.collect()


def start_gc():
    print('Now is starting garbage collecting !!!!!!!!')
    learning_thread = threading.Thread(target=auto_garbage_collect, daemon=True)
    learning_thread.start()
