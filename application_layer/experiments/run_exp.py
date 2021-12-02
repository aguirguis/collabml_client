#!/usr/bin/env python3
# coding: utf-8
#This library include functions to run experiments
import os
import torch
from multiprocessing import Process
homedir = os.path.expanduser("~")
projectdir = os.path.join(homedir, "swift_playground/application_layer")
execfile = os.path.join(projectdir,"main.py")
logdir = os.path.join(projectdir,"logs")

def empty_gpu():
    torch.cuda.empty_cache()
    os.system(f"pkill -f 'python3 {execfile}'")

def run_models_exp(batch_size, models, freeze_idxs, CPU=False):
    #Compare the performance of Vanilla and Split with different models on both GPU and CPU on the client side
    #The default BW here is 1Gbps
    assert len(models) == len(freeze_idxs)
    for model, freeze_idx in zip(models, freeze_idxs):
        empty_gpu()
        #run vanilla
        os.system(f'python3 {execfile} --dataset imagenet --model {model} --num_epochs 1 --batch_size {batch_size}\
		 --freeze --freeze_idx {freeze_idx} {"--cpuonly" if CPU else ""}\
		 > {logdir}/models_exp/vanilla_{model}_bs{batch_size}_{"cpu" if CPU else "gpu"}')
        empty_gpu()
        #run split
        #TWO IMPORTANT NOTES HERE: (1) we use the intermediate server for this experiment, (2) we set server batch size to 200 always
        os.system(f'python3 {execfile} --dataset imagenet --model my{model} --num_epochs 1 --batch_size {batch_size}\
                 --freeze --freeze_idx {freeze_idx} --use_intermediate {"--cpuonly" if CPU else ""}\
		 > {logdir}/models_exp/split_{model}_bs{batch_size}_{"cpu" if CPU else "gpu"}')

def run_bw_exp(BW):
    #Compare the performance of Vanilla and Split with different bandwidth values only with GPUs on the client side
    #The default parameters are: model=Alexnet, freeze_idx=17, batch_size=2000
    wondershaper_exec = os.path.join(homedir,"wondershaper","wondershaper")
    for bw in BW:
        os.system(f'{wondershaper_exec} -c -a eth0')
        os.system(f'{wondershaper_exec} -a eth0 -d {bw} -u {bw}')
        empty_gpu()
        #run vanilla
        os.system(f'python3 {execfile} --dataset imagenet --model alexnet --num_epochs 1 --batch_size 2000\
                 --freeze --freeze_idx 17 > {logdir}/bw_exp/vanilla_{bw/1024}')
        empty_gpu()
        #run split
        os.system(f'python3 {execfile} --dataset imagenet --model myalexnet --num_epochs 1 --batch_size 2000\
                 --freeze --freeze_idx 17 --use_intermediate > {logdir}/bw_exp/split_{bw/1024}')
    #Back to the default BW (1Gbps)
    os.system(f'{wondershaper_exec} -c -a eth0')
    os.system(f'{wondershaper_exec} -a eth0 -d {1024*1024} -u {1024*1024}')

def _run_split(process_idx, batch_size, num_processes):
    #This helper function runs one ML request (the total number of requests is specified in num_processes)
    models_dict={0:("myalexnet",17), 1:("myresnet18",11), 2:("myresnet50",21), 3:("myvgg11",25), 4:("myvgg19",36),5:("mydensenet121",19)}
    model, freeze_idx = models_dict[process_idx]
    os.system(f'python3 {execfile} --dataset imagenet --model {model} --num_epochs 1 --batch_size {batch_size}\
                 --freeze --freeze_idx {freeze_idx} --use_intermediate > {logdir}/multitenant_exp/process_{process_idx+1}_of_{num_processes}_bs{batch_size}')

def run_scalability_multitenants(max_tenants, batch_sizes):
    #Test the scalability of our system (i.e., split) with multi-tenants and different batch sizes
    #Default parameters: (model, freeze_idx)=(multiple values), BW=1Gbps
    for t in range(max_tenants):
        num_tenant=t+1
        for bsz in batch_sizes:
            process = []
            empty_gpu()
            for i in range(num_tenant):
                p = Process(target=_run_split, args=(i,bsz,num_tenant))
                p.start()
                process.append(p)
            for p in process:
                p.join()


if __name__ == '__main__':
###################################EXP 1: MODELS EXP#############################################
#    models=['resnet18', 'resnet50', 'vgg11','vgg19', 'alexnet', 'densenet121']
#    freeze_idxs=[11, 21, 25, 36, 17, 19]
#    bsz = 2000		#This is the largest number I can get that fits in the client GPU
#    run_models_exp(bsz, models, freeze_idxs) #GPU on the client side
#    run_models_exp(bsz, models, freeze_idxs, CPU=True) #CPU on the client side
    #The same experiment but with extremely big batch size
#    bsz = 8000		#This fails with vanila GPU (but should hopefully work on)
#    run_models_exp(bsz, models, freeze_idxs) #GPU on the client side
#    run_models_exp(bsz, models, freeze_idxs, CPU=True) #CPU on the client side
#################################################################################################
###################################EXP 2: BW EXP#################################################
#    BW = [50*1024, 100*1024, 500*1024, 1024*1024, 2*1024*1024]
#    run_bw_exp(BW[:1])
#################################################################################################
###################################EXP 3: Scalability with multi-tenants EXP#####################
#    max_tenants = 6
#    batch_sizes = [1000,2000,3000,4000,6000,8000]
#    run_scalability_multitenants(max_tenants, batch_sizes[1:2])
#################################################################################################
