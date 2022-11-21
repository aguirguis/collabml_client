#!/usr/bin/env python3

from multiprocessing import Process, Manager, Pool
import multiprocessing as mp
import os
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import torch

homedir = os.path.expanduser("~")
projectdir = os.path.join(homedir, "collabml_client/application_layer")
execfile = os.path.join(projectdir,"main.py")
logdir = os.path.join(projectdir,"logs")

num_processes = 3
batch_size = 2000
special_dir="multitenant_ALL_IN_COS"
split_choice="automatic"


INITIAL_TIME = time.time()

def empty_gpu():
    #time.sleep(10)
    #print("HERE")
    torch.cuda.empty_cache()
    os.system(f"pkill -f 'python3 {execfile}'")
    #print("HERE2")
    #time.sleep(10)
    #print("HERE3")

def _run_vanilla(process_idx):
    #This helper function runs one ML request (the total number of requests is specified in num_processes)
    if (process_idx+process_idx/num_processes)%2 == 0:
        time.sleep(2)
    print(process_idx)
    models_dict={0:("alexnet",17)}
    model, freeze_idx = models_dict[0]	#[process_idx]
    
    dir = special_dir
    file_name = logdir+'/'+dir+'_'+split_choice+'/process_'+str(process_idx+1)+'_of_'+str(num_processes)+'_bs'+str(batch_size)
    os.system(f'python3 {execfile} --dataset imagenet --model {model} --num_epochs 1 --batch_size {batch_size} --freeze --freeze_idx {freeze_idx} > {file_name}')
    
    time_to_write = time.time()
    lines = open(file_name, 'r').readlines()
    lines[-1] = "The whole process took " + str(time.time()-INITIAL_TIME) + " seconds"
    open(file_name, 'w').writelines(lines)
    #print("WRITE: ", time.time()-time_to_write)
    #time.sleep(2)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    empty_gpu()

    #pool = Pool(processes=2, maxtasksperchild=1)
    pool = Pool(processes=2)
    #pool = ThreadPoolExecutor(3)
    keys = range(num_processes)

    #print("INIT TIME :", time.time()-INITIAL_TIME)
    result = pool.map(_run_vanilla, keys)

    #manager = Manager()
    #l = manager.list(range(nb_process))



#    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
#        for process_idx in range(nb_process):
#            print(process_idx)
#            future = executor.submit(_run_vanilla, process_idx, 2000, nb_process)
#            #future.result()
#            #print("FINISHED")

#    #while l not empty:
#    process_idx = 0
#    p1 = Process(target=_run_vanilla, args=(process_idx, 2000, nb_process))
#    p1.start()
#
#    p1.join()
#    print("FINISHED HERE")

