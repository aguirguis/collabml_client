#!/usr/bin/env python3
# coding: utf-8
#This library include functions to run experiments
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
    time.sleep(10)
    #print("HERE")
    torch.cuda.empty_cache()
    os.system(f"pkill -f 'python3 {execfile}'")
    #print("HERE2")
    time.sleep(10)
    #print("HERE3")

def run_alexnet_all_indexes(batch_size, freeze_idx, CPU=False, bw=1024):
    wondershaper_exec = os.path.join(homedir,"wondershaper","wondershaper")
    os.system(f'{wondershaper_exec} -c -a eth0')
    m_bw = bw * 1024
    os.system(f'{wondershaper_exec} -a eth0 -d {m_bw} -u {m_bw}')
    #The default BW here is 1Gbps

    for split_idx in range(freeze_idx):
    #for split_idx in range(0,7):
        empty_gpu()
        #run split
        os.system(f'python3 {execfile} --dataset imagenet --model myalexnet --num_epochs 1 --batch_size {batch_size}\
                 --freeze --freeze_idx {freeze_idx} --split_idx {split_idx} --use_intermediate --split_choice manual {"--cpuonly" if CPU else ""}\
                > {logdir}/alexnet/test_split_alexnet_bs{batch_size}_split_{split_idx}_bw{m_bw}_{"cpu" if CPU else "gpu"}')
        time.sleep(20)

    os.system(f'{wondershaper_exec} -c -a eth0')
    os.system(f'{wondershaper_exec} -a eth0 -d {1024*1024} -u {1024*1024}')

def run_models_exp_min_split(batch_size, models, freeze_idxs, CPU=False, bw=1024):
    wondershaper_exec = os.path.join(homedir,"wondershaper","wondershaper")
    os.system(f'{wondershaper_exec} -c -a eth0')
    m_bw = bw * 1024
    os.system(f'{wondershaper_exec} -a eth0 -d {m_bw} -u {m_bw}')
    #The default BW here is 1Gbps

    assert len(models) == len(freeze_idxs)

    for model, freeze_idx in zip(models, freeze_idxs):
        empty_gpu()
        os.system(f'python3 {execfile} --dataset imagenet --model my{model} --num_epochs 1 --batch_size {batch_size}\
                 --freeze --freeze_idx {freeze_idx} --use_intermediate --split_choice to_min {"--cpuonly" if CPU else ""}\
                > {logdir}/models_exp/test_min_{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU else "gpu"}')
    
    os.system(f'{wondershaper_exec} -c -a eth0')
    os.system(f'{wondershaper_exec} -a eth0 -d {1024*1024} -u {1024*1024}')

def run_models_exp_freeze_split(batch_size, models, freeze_idxs, CPU=False, bw=1024):
    wondershaper_exec = os.path.join(homedir,"wondershaper","wondershaper")
    os.system(f'{wondershaper_exec} -c -a eth0')
    m_bw = bw*1024
    os.system(f'{wondershaper_exec} -a eth0 -d {m_bw} -u {m_bw}')
    #The default BW here is 1Gbps

    assert len(models) == len(freeze_idxs)

    for model, freeze_idx in zip(models, freeze_idxs):
        empty_gpu()
        #run split
        os.system(f'python3 {execfile} --dataset imagenet --model my{model} --num_epochs 1 --batch_size {batch_size}\
                 --freeze --freeze_idx {freeze_idx} --use_intermediate {"--cpuonly" if CPU else ""}\
                > {logdir}/models_exp/test_split_{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU else "gpu"}')
        #run freeze
        empty_gpu()
        os.system(f'python3 {execfile} --dataset imagenet --model my{model} --num_epochs 1 --batch_size {batch_size}\
                 --freeze --freeze_idx {freeze_idx} --use_intermediate --split_choice to_freeze {"--cpuonly" if CPU else ""}\
                > {logdir}/models_exp/test_freeze_{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU else "gpu"}')
    
    os.system(f'{wondershaper_exec} -c -a eth0')
    os.system(f'{wondershaper_exec} -a eth0 -d {1024*1024} -u {1024*1024}')


def run_models_exp(batch_size, models, freeze_idxs, CPU=False, bw=1024, dataset='imagenet'):
    #Compare the performance of Vanilla and Split with different models on both GPU and CPU on the client side
    #The default BW here is 1Gbps
    wondershaper_exec = os.path.join(homedir,"wondershaper","wondershaper")
    os.system(f'{wondershaper_exec} -c -a eth0')
    m_bw = bw*1024
    os.system(f'{wondershaper_exec} -a eth0 -d {m_bw} -u {m_bw}')
    #The default BW here is 1Gbps
    assert len(models) == len(freeze_idxs)
    for model, freeze_idx in zip(models, freeze_idxs):
        empty_gpu()
        #run vanilla
        os.system(f'python3 {execfile} --dataset {dataset} --model {model} --num_epochs 1 --batch_size {batch_size}\
		 --freeze --freeze_idx {freeze_idx} {"--cpuonly" if CPU else ""}\
		 > {logdir}/models_exp_{dataset}/vanilla_{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU else "gpu"}')
        empty_gpu()
        #run split
        #TWO IMPORTANT NOTES HERE: (1) we use the intermediate server for this experiment, (2) we set server batch size to 200 always
        os.system(f'python3 {execfile} --dataset {dataset} --model my{model} --num_epochs 1 --batch_size {batch_size}\
                 --freeze --freeze_idx {freeze_idx} --use_intermediate {"--cpuonly" if CPU else ""}\
		> {logdir}/models_exp_{dataset}/test_split_{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU else "gpu"}')

    os.system(f'{wondershaper_exec} -c -a eth0')
    os.system(f'{wondershaper_exec} -a eth0 -d {1024*1024} -u {1024*1024}')

def vit_run_models_exp(batch_size, models, freeze_idxs, CPU=False, bw=1024, dataset='imagenet'):
    #Compare the performance of Vanilla and Split with different models on both GPU and CPU on the client side
    #The default BW here is 1Gbps
    assert len(models) == len(freeze_idxs)
    wondershaper_exec = os.path.join(homedir,"wondershaper","wondershaper")
    os.system(f'{wondershaper_exec} -c -a eth0')
    m_bw = bw*1024
    os.system(f'{wondershaper_exec} -a eth0 -d {m_bw} -u {m_bw}')
    for model, freeze_idx in zip(models, freeze_idxs):
        empty_gpu()
        #run vanilla
        os.system(f'python3 {execfile} --dataset {dataset} --model {model} --num_epochs 1 --batch_size {batch_size}\
		 --freeze --freeze_idx {freeze_idx} {"--cpuonly" if CPU else ""}\
		 > {logdir}/vit_models_exp/vanilla_{model}_bs{batch_size}_{"cpu" if CPU else "gpu"}')
        empty_gpu()
        #run split
        #TWO IMPORTANT NOTES HERE: (1) we use the intermediate server for this experiment, (2) we set server batch size to 200 always
        os.system(f'python3 {execfile} --dataset {dataset} --model my{model} --num_epochs 1 --batch_size {batch_size}\
                 --freeze --freeze_idx {freeze_idx} --use_intermediate {"--cpuonly" if CPU else ""}\
		 > {logdir}/vit_models_exp_{dataset}/test_split_{model}_bs{batch_size}_{"cpu" if CPU else "gpu"}')

    os.system(f'{wondershaper_exec} -c -a eth0')
    os.system(f'{wondershaper_exec} -a eth0 -d {1024*1024} -u {1024*1024}')

def run_swift_in_and_out(batch_size, models, freeze_idxs):
    #Compare the performance of running inside and outside of Swift (with different models)
    #The default BW here is 1Gbps
    assert len(models) == len(freeze_idxs)
    for model, freeze_idx in zip(models, freeze_idxs):
        empty_gpu()
        #run computation inside Swift
        os.system(f'python3 {execfile} --dataset imagenet --model my{model} --num_epochs 1 --batch_size {batch_size}\
                 --freeze --freeze_idx {freeze_idx}\
                 > {logdir}/swift_inout/swiftin_{model}_bs{batch_size}_"gpu"')
#        empty_gpu()
        #run computation out of Swift
        #ONE IMPORTANT NOTES HERE: we set server batch size to 200 always
#        os.system(f'python3 {execfile} --dataset imagenet --model my{model} --num_epochs 1 --batch_size {batch_size}\
#                 --freeze --freeze_idx {freeze_idx} --use_intermediate\
#                 > {logdir}/swift_inout/swiftout_{model}_bs{batch_size}_"gpu"')

def run_bw_exp_freeze_split(BW, model, freeze_idx):
    wondershaper_exec = os.path.join(homedir,"wondershaper","wondershaper")
    for bw in BW:
        os.system(f'{wondershaper_exec} -c -a eth0')
        os.system(f'{wondershaper_exec} -a eth0 -d {bw} -u {bw}')
        #empty_gpu()
        #run freeze
        os.system(f'python3 {execfile} --dataset imagenet --model my{model} --num_epochs 1 --batch_size 8000\
                 --freeze --freeze_idx {freeze_idx} --use_intermediate --split_choice to_freeze > {logdir}/bw_exp/freeze_{bw/1024}_{model}')
        empty_gpu()
        #run split
        os.system(f'python3 {execfile} --dataset imagenet --model my{model} --num_epochs 1 --batch_size 8000\
                 --freeze --freeze_idx {freeze_idx} --use_intermediate > {logdir}/bw_exp/split_{bw/1024}_{model}')
    #Back to the default BW (1Gbps)
    os.system(f'{wondershaper_exec} -c -a eth0')
    os.system(f'{wondershaper_exec} -a eth0 -d {1024*1024} -u {1024*1024}')

def run_bw_exp(BW, model, freeze_idx, dataset='imagenet'):
    #Compare the performance of Vanilla and Split with different bandwidth values only with GPUs on the client side
    #The default parameters are: model=Alexnet, freeze_idx=17, batch_size=2000
    wondershaper_exec = os.path.join(homedir,"wondershaper","wondershaper")
    for bw in BW:
        os.system(f'{wondershaper_exec} -c -a eth0')
        os.system(f'{wondershaper_exec} -a eth0 -d {bw} -u {bw}')
        #empty_gpu()
        #run vanilla
        os.system(f'python3 {execfile} --dataset {dataset} --model {model} --num_epochs 1 --batch_size 8000\
                 --freeze --freeze_idx {freeze_idx} > {logdir}/bw_exp_{dataset}/vanilla_{bw/1024}_{model}')
        empty_gpu()
        #run split
        os.system(f'python3 {execfile} --dataset {dataset} --model my{model} --num_epochs 1 --batch_size 8000\
                 --freeze --freeze_idx {freeze_idx} --use_intermediate > {logdir}/bw_exp_{dataset}/split_{bw/1024}_{model}')
    #Back to the default BW (1Gbps)
    os.system(f'{wondershaper_exec} -c -a eth0')
    os.system(f'{wondershaper_exec} -a eth0 -d {1024*1024} -u {1024*1024}')

def _run_split(process_idx, batch_size, num_processes, special_dir="", split_choice="automatic"):
    #This helper function runs one ML request (the total number of requests is specified in num_processes)
    #models_dict={0:("myalexnet",17), 1:("myresnet18",11), 2:("myresnet50",21), 3:("myvgg11",25), 4:("myvgg19",36),5:("mydensenet121",20)}
    #models_dict={0:("myalexnet",17), 1:("mydensenet121",20)}
    print("Run split ", split_choice)
    models_dict={0:("myvgg19",36)}
    model, freeze_idx = models_dict[process_idx%len(models_dict)]
    dir = 'multitenant_exp' if special_dir=="" else special_dir
    os.system(f'python3 {execfile} --dataset imagenet --model {model} --num_epochs 1 --batch_size {batch_size}\
                 --freeze --freeze_idx {freeze_idx} --use_intermediate --split_choice {split_choice} > {logdir}/{dir}_{split_choice}/process_{process_idx+1}_of_{num_processes}_bs{batch_size}')

def _run_vanilla(process_idx, batch_size, num_processes,special_dir=""):
    #This helper function runs one ML request (the total number of requests is specified in num_processes)
    #models_dict={0:("alexnet",17), 1:("resnet18",11), 2:("resnet50",21), 3:("vgg11",25), 4:("vgg19",36),5:("densenet121",20)}
    models_dict={0:("alexnet",17)}
    model, freeze_idx = models_dict[0]	#[process_idx]
    dir = 'vanilla_run' if special_dir=="" else special_dir
    os.system(f'python3 {execfile} --dataset imagenet --model {model} --num_epochs 1 --batch_size {batch_size}\
                 --freeze --freeze_idx {freeze_idx} > {logdir}/{dir}/process_{process_idx+1}_of_{num_processes}_bs{batch_size}')

def run_scalability_multitenants(max_tenants, batch_sizes, target="split", split_choice="automatic", bw=1024):
    #Test the scalability of our system (i.e., split) with multi-tenants and different batch sizes
    #Default parameters: (model, freeze_idx)=(multiple values), BW=1Gbps
    wondershaper_exec = os.path.join(homedir,"wondershaper","wondershaper")
    os.system(f'{wondershaper_exec} -c -a eth0')
    m_bw = bw * 1024
    os.system(f'{wondershaper_exec} -a eth0 -d {m_bw} -u {m_bw}')
    #for t in range(0, max_tenants, 2):
    for t in range(1,2):
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
##################################EXP 0: MODELAS EXP MIN SPLIT#############################################
#    BWS = [1024]#, 50]
#
#    for bw in BWS:
#        #models = ['resnet18', 'resnet50', 'vgg11', 'vgg19', 'alexnet', 'densenet121']
#        models = ['vgg19']
#        #models = ['alexnet']
#        #freeze_idxs = [11, 21, 25, 36, 17, 20]
#        freeze_idxs = [36]
#
#        #bsz = 8000
#        bsz = 128
#
        #run_models_exp_min_split(bsz, models, freeze_idxs, bw=bw)  # GPU on the client side

        #run_models_exp_min_split(bsz, models, freeze_idxs, CPU=True, bw=bw)  # CPU on the client side

        #bsz = 2000

        #run_models_exp_min_split(bsz, models, freeze_idxs, bw=bw)  # GPU on the client side

        #run_models_exp_min_split(bsz, models, freeze_idxs, CPU=True, bw=bw)  # CPU on the client side
##################################EXP 0: MODELS EXP FREEZE AND SPLIT#############################################
#    for bw in BWS:
#        models=['resnet18', 'resnet50', 'vgg11','vgg19', 'alexnet', 'densenet121']
#
#        freeze_idxs=[11, 21, 25, 36, 17, 20]
#
#        #bsz = 8000
#        bsz = 128
#
#        run_models_exp_freeze_split(bsz, models, freeze_idxs, bw=bw) #GPU on the client side
#
#        run_models_exp_freeze_split(bsz, models, freeze_idxs, CPU=True, bw=bw) #CPU on the client side

        #bsz = 1000

        #run_models_exp_freeze_split(bsz, models, freeze_idxs, bw=bw) #GPU on the client side

        #run_models_exp_freeze_split(bsz, models, freeze_idxs, CPU=True, bw=bw) #CPU on the client side


#    BW = [50*1024, 100*1024, 500*1024, 1024*1024, 2*1024*1024, 3*1024*1024,5*1024*1024, 10*1024*1024, 15*1024*1024]
#    run_bw_exp_freeze_split(BW, "alexnet", 17)
##################################EXP 0: MODELS EXP MIN SPLIT#############################################
    #for bw in BWS:
        #bsz = 8000
        #run_alexnet_all_indexes(bsz, 17, bw=bw)
        #run_alexnet_all_indexes(bsz, 17, CPU=True, bw=bw)
        #bsz = 2000
        #run_alexnet_all_indexes(bsz, 17, bw=bw)
        #run_alexnet_all_indexes(bsz, 17, CPU=True, bw=bw)

##################################EXP 1: MODELS EXP#############################################
    models=['resnet18', 'resnet50', 'vgg11','vgg19', 'alexnet', 'densenet121']
#    models=['vit']
    freeze_idxs=[11, 21, 25, 36, 17, 20]
#    freeze_idxs=[17]
    bsz = 250
    dataset = 'plantleave'
    #bsz = 2000		#This is the largest number I can get that fits in the client GPU
    run_models_exp(bsz, models, freeze_idxs, dataset=dataset)  # GPU on the client side
    run_models_exp(bsz, models, freeze_idxs, CPU=True, dataset=dataset)  # GPU on the client side
#    bsz = 8000
#    vit_run_models_exp(bsz, models, freeze_idxs) #GPU on the client side
#    vit_run_models_exp(bsz, models, freeze_idxs, CPU=True)
#    #run_models_exp(bsz, models, freeze_idxs) #GPU on the client side
        #:run_models_exp(bsz, models, freeze_idxs, CPU=True, bw=bw) #CPU on the client side
################################################################################################
##################################EXP 2: BW EXP#################################################
    BW = [50*1024, 100*1024, 500*1024, 1024*1024, 2*1024*1024, 3*1024*1024,5*1024*1024, 10*1024*1024, 12*1024*1024]
#    #BW = [50*1024]
#    BW = [1024*1024, 12*1024*1024]
##    run_bw_exp(BW, "vit", 17)

    run_bw_exp(BW, "alexnet", 17, dataset)
##################################################################################################
###################################EXP 3: Scalability with multi-tenants EXP#####################
    #max_tenants = 11
    #batch_sizes = [1000,4000]
    #batch_sizes = [1000]
    #run_scalability_multitenants(max_tenants, batch_sizes[:1], split_choice='to_min')
    #run_scalability_multitenants(max_tenants, batch_sizes[:1], split_choice='to_max')
    #run_scalability_multitenants(max_tenants, batch_sizes[:1])
    #run_scalability_multitenants(max_tenants, batch_sizes[:1],"vanilla")
    #run_scalability_multitenants(max_tenants, batch_sizes[1:])
###############################################################################################
##############################EXP 4: Data reduction with different batch sizes#################
####Not really a complete experiment, yet this is useful to compare vanilla to split###########
#Note that: for this experiment only, I'm setting the server batch size to 1000 (in order to overload the server memory)
#    batch_sizes = [1000,2000,3000,4000, 5000, 6000,7000,8000, 10000,12000,14000]
#    batch_sizes = [2000,4000,6000,8000,10000,12000]
#    for bsz in batch_sizes:
#        empty_gpu()
#        _run_vanilla(0, bsz, 1)
#        empty_gpu()
#        _run_split(0, bsz, 1,special_dir='withbatchAdatp_exp_afterfix_serverb200')		#note that: the dir without batch adaptation is: 'batchAdatp_exp'
###############################################################################################
##############################EXP 5: run computation inside and outside of Swift#################
#    models=['resnet18', 'resnet50', 'alexnet', 'densenet121']
#    freeze_idxs=[11, 21, 17, 20]
#    bsz = 2000         #This is the largest number I can get that fits in the client GPU
#    run_swift_in_and_out(bsz, models, freeze_idxs)
