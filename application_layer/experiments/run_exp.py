
#!/usr/bin/env python3
# coding: utf-8
#This library include functions to run experiments
import os
import time
import torch
from multiprocessing import Process
homedir = os.path.expanduser("~")
projectdir = os.path.join(homedir, "collabml_client/application_layer")
execfile = os.path.join(projectdir,"main.py")
logdir = os.path.join(projectdir,"logs")

def empty_gpu():
    time.sleep(10)
    torch.cuda.empty_cache()
    os.system(f"pkill -f 'python3 {execfile}'")
    time.sleep(10)

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

def run_bw_exp(BW, model, freeze_idx):
    #Compare the performance of Vanilla and Split with different bandwidth values only with GPUs on the client side
    #The default parameters are: model=Alexnet, freeze_idx=17, batch_size=2000
    wondershaper_exec = os.path.join(homedir,"wondershaper","wondershaper")
    for bw in BW:
        os.system(f'{wondershaper_exec} -c -a eth0')
        os.system(f'{wondershaper_exec} -a eth0 -d {bw} -u {bw}')
        empty_gpu()
        #run vanilla
        os.system(f'python3 {execfile} --dataset imagenet --model {model} --num_epochs 1 --batch_size 200\
                 --freeze --freeze_idx {freeze_idx} > {logdir}/bw_exp/vanilla_{bw/1024}_{model}')
#        empty_gpu()
        #run split
        os.system(f'python3 {execfile} --dataset imagenet --model my{model} --num_epochs 1 --batch_size 200\
                 --freeze --freeze_idx {freeze_idx} --use_intermediate > {logdir}/bw_exp/split_{bw/1024}_{model}')
    #Back to the default BW (1Gbps)
    os.system(f'{wondershaper_exec} -c -a eth0')
    os.system(f'{wondershaper_exec} -a eth0 -d {1024*1024} -u {1024*1024}')

def _run_split(process_idx, batch_size, num_processes, special_dir=""):
    #This helper function runs one ML request (the total number of requests is specified in num_processes)
    models_dict={0:("myalexnet",17), 1:("myresnet18",11), 2:("myresnet50",21), 3:("myvgg11",25), 4:("myvgg19",36),5:("mydensenet121",20)}
    model, freeze_idx = models_dict[process_idx%len(models_dict)]
    dir = 'multitenant_exp' if special_dir=="" else special_dir
    os.system(f'python3 {execfile} --dataset imagenet --model {model} --num_epochs 1 --batch_size {batch_size}\
                 --freeze --freeze_idx {freeze_idx} --use_intermediate > {logdir}/{dir}/process_{process_idx+1}_of_{num_processes}_bs{batch_size}')

def _run_vanilla(process_idx, batch_size, num_processes,special_dir=""):
    #This helper function runs one ML request (the total number of requests is specified in num_processes)
    models_dict={0:("alexnet",17), 1:("resnet18",11), 2:("resnet50",21), 3:("vgg11",25), 4:("vgg19",36),5:("densenet121",20)}
    model, freeze_idx = models_dict[0]	#[process_idx]
    dir = 'vanilla_run' if special_dir=="" else special_dir
    os.system(f'python3 {execfile} --dataset imagenet --model {model} --num_epochs 1 --batch_size {batch_size}\
                 --freeze --freeze_idx {freeze_idx} > {logdir}/{dir}/process_{process_idx+1}_of_{num_processes}_bs{batch_size}')

def run_scalability_multitenants(max_tenants, batch_sizes, target="split"):
    #Test the scalability of our system (i.e., split) with multi-tenants and different batch sizes
    #Default parameters: (model, freeze_idx)=(multiple values), BW=1Gbps
    for t in range(11, max_tenants, 2):
        num_tenant=t+1
        for bsz in batch_sizes:
            process = []
            empty_gpu()
            for i in range(num_tenant):
                if target == "vanilla":
                    p = Process(target=_run_vanilla, args=(i,bsz,num_tenant,"multitenant_exp_vanilla_notraining"))
                else:
                    p = Process(target=_run_split, args=(i,bsz,num_tenant, "multitenant_exp_notraining"))
                p.start()
                process.append(p)
            for p in process:
                p.join()

if __name__ == '__main__':
###################################EXP 1: MODELS EXP#############################################
#    #models=['resnet18', 'resnet50', 'vgg11','vgg19', 'alexnet', 'densenet121']
#    models=['vit']
#    #freeze_idxs=[11, 21, 25, 36, 17, 20]
#    freeze_idxs=[17]
#    #bsz = 2000		#This is the largest number I can get that fits in the client GPU
#    bsz = 200
#    run_models_exp(bsz, models, freeze_idxs) #GPU on the client side
#    run_models_exp(bsz, models, freeze_idxs, CPU=True) #CPU on the client side
#   #The same experiment but with extremely big batch size
#    bsz = 1000		#This fails with vanila GPU (but should hopefully work on)
#    run_models_exp(bsz, models, freeze_idxs) #GPU on the client side
#    run_models_exp(bsz, models, freeze_idxs, CPU=True) #CPU on the client side
#################################################################################################
###################################EXP 2: BW EXP#################################################
#    BW = [50*1024, 100*1024, 500*1024, 1024*1024, 2*1024*1024, 3*1024*1024,5*1024*1024, 10*1024*1024, 15*1024*1024]
    BW = [1024*1024]
    #BW = [1024*1024, 12*1024*1024]
    run_bw_exp(BW, "vit", 17)
#    run_bw_exp(BW, "alexnet", 17)
#################################################################################################
###################################EXP 3: Scalability with multi-tenants EXP#####################
#    max_tenants = 11
#    batch_sizes = [1000,4000]
#    run_scalability_multitenants(max_tenants, batch_sizes[:1])
#    run_scalability_multitenants(max_tenants, batch_sizes[:1],"vanilla")
#    run_scalability_multitenants(max_tenants, batch_sizes[1:])
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
