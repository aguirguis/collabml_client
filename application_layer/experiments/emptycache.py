
import os
import time
import torch
from time import sleep
from multiprocessing import Process
homedir = os.path.expanduser("~")
projectdir = os.path.join(homedir, "collabml_client/application_layer")
execfile = os.path.join(projectdir,"main.py")
logdir = os.path.join(projectdir,"logs")

def empty_gpu():
    #time.sleep(10)
    #print("HERE")
    torch.cuda.empty_cache()
    os.system(f"pkill -f 'python3 {execfile}'")
    #print("HERE2")
    time.sleep(10)

if __name__ == '__main__':
    empty_gpu()
