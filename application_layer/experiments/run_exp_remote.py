#!/usr/bin/env python3
# coding: utf-8
#This library include functions to run experiments
import os
import time
import torch
from time import sleep
from multiprocessing import Process
import argparse
homedir = os.path.expanduser("~")
projectdir = os.path.join(homedir, "collabml_client/application_layer")
execfile = os.path.join(projectdir,"main.py")
logdir = os.path.join(projectdir,"logs")

def empty_gpu():
    time.sleep(10)
    torch.cuda.empty_cache()
    os.system(f"pkill -f 'python3 {execfile}'")
    time.sleep(10)

models_dict={0:("myalexnet",17), 1:("myresnet18",11), 2:("myresnet50",21), 3:("myvgg11",25), 4:("myvgg19",36),5:("mydensenet121",20)}

def _run_split(process_idx, batch_size, num_processes, model_idx, special_dir="", split_choice="automatic", split_idx=9):
    #This helper function runs one ML request (the total number of requests is specified in num_processes)
    print("Run split ", split_choice)
    model, freeze_idx = models_dict[model_idx]
    dir = 'multitenant_exp' if special_dir=="" else special_dir
    if split_choice != 'manual':
        os.system(f'python3 {execfile} --dataset imagenet --model {model} --num_epochs 1 --batch_size {batch_size}\
                 --freeze --freeze_idx {freeze_idx} --use_intermediate --split_choice {split_choice} > {logdir}/{dir}_{split_choice}/process_{process_idx+1}_of_{num_processes}_bs{batch_size}')
    else:
        os.system(f'python3 {execfile} --dataset imagenet --model {model} --num_epochs 1 --batch_size {batch_size}\
                                 --freeze --freeze_idx {freeze_idx} --use_intermediate --split_choice {split_choice} --split_idx {split_idx} > {logdir}/{dir}_{split_choice}/process_{process_idx+1}_of_{num_processes}_bs{batch_size}_splitidx_{split_idx}')

def _run_vanilla(process_idx, batch_size, num_processes, model_idx, special_dir=""):
    #This helper function runs one ML request (the total number of requests is specified in num_processes)
    model, freeze_idx = models_dict[model_idx]	#[process_idx]
    dir = 'vanilla_run' if special_dir=="" else special_dir
    os.system(f'python3 {execfile} --dataset imagenet --model {model} --num_epochs 1 --batch_size {batch_size}\
                 --freeze --freeze_idx {freeze_idx} > {logdir}/{dir}/process_{process_idx+1}_of_{num_processes}_bs{batch_size}')

def run_scalability_multitenants(max_tenants, batch_sizes, target="split", split_choice="automatic", bw=1024):
    #Test the scalability of our system (i.e., split) with multi-tenants and different batch sizes
    #Default parameters: (model, freeze_idx)=(multiple values), BW=1Gbps
    #for t in range(0, max_tenants, 2):
    for t in range(2,3):
        num_tenant=t+1
        for bsz in batch_sizes:
            process = []
            empty_gpu()
            for i in range(num_tenant):
                if target == "vanilla":
                    p = Process(target=_run_vanilla, args=(i,bsz,num_tenant,"multitenant_exp_vanilla"))
                else:
                    p = Process(target=_run_split, args=(i,bsz,num_tenant, "multitenant_exp", split_choice))
                    #p = Process(target=_run_split, args=(i,bsz,num_tenant, "multitenant_exp_notraining_new", split_choice))
                p.start()
                process.append(p)
            for p in process:
                p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment multi tenant')

    parser.add_argument('--batch_size', default=1000, type=int, help='batch size for dataloader')

    parser.add_argument('--target', default='split', type=str, help='split or vanilla')

    parser.add_argument('--split_choice', default='automatic', type=str, help='How to choose split_idx (manual, automatic, to_freeze, to_min, to_max)')
    
    parser.add_argument('--split_idx', default=9, type=int, help='')

    parser.add_argument('--num_tenant', default=6, type=int, help='total number of tenants for this experiment')

    parser.add_argument('--process_idx', default=0, type=int, help='process id')

    parser.add_argument('--model_idx', default=0, type=int, help=f'model id for {models_dict}')

    args = parser.parse_args()


    wondershaper_exec = os.path.join(homedir,"wondershaper","wondershaper")
    os.system(f'{wondershaper_exec} -c -a eth0')
    #bw = 1024
    #m_bw = bw * 1024 * 1024
    #os.system(f'{wondershaper_exec} -a eth0 -d {m_bw} -u {m_bw}')
    
    batch_size = args.batch_size
    target = args.target
    split_choice = args.split_choice
    num_tenant = args.num_tenant
    process_idx = args.process_idx
    model_idx = args.model_idx
    split_idx = args.split_idx
    

    curr_exp_dir = os.path.join(f"{logdir}", f"multitenant_exp_{target}_{split_choice}")
    if not os.path.exists(curr_exp_dir):
        os.makedirs(curr_exp_dir)

    p = Process(target=_run_split, args=(process_idx, batch_size, num_tenant, model_idx, f"multitenant_exp_{target}", split_choice, split_idx))
    p.start()
    p.join()
    #run_scalability_multitenants(max_tenants, batch_sizes[:1], split_choice='to_min')
    #run_scalability_multitenants(max_tenants, batch_sizes[:1], split_choice='to_max')
    #run_scalability_multitenants(max_tenants, batch_sizes[:1])
    #run_scalability_multitenants(max_tenants, batch_sizes[:1],"vanilla")
    #run_scalability_multitenants(max_tenants, batch_sizes[1:])
