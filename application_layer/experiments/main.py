#!/usr/bin/env python3
# coding: utf-8
#This is the main script to parse the log files and plot the paper figures
import os
import numpy as np
from myparser import *
from myplotter import *
homedir = os.path.expanduser("~")
projectdir = os.path.join(homedir, "swift_playground/application_layer")
logdir = os.path.join(projectdir,"logs")
hatches = ['//','\\','x','o',"*","."]
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1)), 'solid', '--']
#This will be a monolithic script with no functions
#It should be somehow the mirror of run_exp.py
#EXP1: models exp
#exp_name="models_exp"
#specific_dir = os.path.join(logdir, exp_name)
#models=['resnet18', 'resnet50', 'vgg11','vgg19', 'alexnet', 'densenet121']
#logs are stored in {specific_dir}/{vanilla_or_split}_{model}_bs{batch_size}_{cpu_or_gpu}
#devs=['gpu','cpu']
#for dev in devs:
#    Y=[]
#    bsz=2000
#    Y.append([os.path.join(specific_dir,f"vanilla_{model}_bs{bsz}_{dev}") for model in models])
#    Y.append([os.path.join(specific_dir,f"split_{model}_bs{bsz}_{dev}") for model in models])
#    bsz=8000
#    Y.append([os.path.join(specific_dir,f"vanilla_{model}_bs{bsz}_{dev}") for model in models])
#    Y.append([os.path.join(specific_dir,f"split_{model}_bs{bsz}_{dev}") for model in models])
#    Y = [get_total_exec_time(filenames) for filenames in Y]
#    sys_legends = ["Vanilla - B=2000","Split - B=2000","Vanilla - B=8000","Split - B=8000"]
#    xtick_labels = [model.title() for model in models]
#    plot_bars(Y, sys_legends, xtick_labels, hatches, "Models", "Execution Time (sec.)", f"results/{exp_name}_{dev}")
##################################################################################################
#EXP2: BW exp
#exp_name="bw_exp"
#specific_dir = os.path.join(logdir, exp_name)
#BW = [50*1024, 100*1024, 500*1024, 1024*1024, 2*1024*1024, 3*1024*1024]
#Y=[]
#Y.append([os.path.join(specific_dir,f"vanilla_{bw/1024}") for bw in BW])
#Y.append([os.path.join(specific_dir,f"split_{bw/1024}") for bw in BW])
#exec_time = [get_total_exec_time(filenames) for filenames in Y]
#sys_legends = ["Vanilla", "Split"]
#xtick_labels = [int(bw/1024) for bw in BW]
#plot_bars(exec_time, sys_legends, xtick_labels, hatches, "Bandwidth (Mbps)", "Execution Time (sec.)", f"results/{exp_name}_exectime")
#######Another thing we want to see is the split index in each case
#split_idxs = get_split_idx(Y[1:])
#plot_bars(split_idxs, sys_legends[1:], xtick_labels, hatches, "Bandwidth (Mbps)", "Split Index", f"results/{exp_name}_splitidx")
#######THe third thing to plot is the output size in each case
#output_sizes = get_output_size(Y[1:])
#plot_bars(output_sizes,sys_legends[1:], xtick_labels, hatches, "Bandwidth (Mbps)", "Transferred Data (MBs)",f"results/{exp_name}_outputsizes")
#################################################################################################
#EXP3: Scalability with multiple tenants exp
exp_name="multitenant_exp"
specific_dir = os.path.join(logdir, exp_name)
batch_sizes = [500, 1000,2000, 4000]
max_tenants = 5
Y = []
for batch_size in batch_sizes:
    times = []
    for t in range(max_tenants):
        num_tenants = t+1
        filenames = [os.path.join(specific_dir,f"process_{idx+1}_of_{num_tenants}_bs{batch_size}") for idx in range(num_tenants)]
        exec_time=get_total_exec_time(filenames)
        if len(exec_time) > 4:
            exec_time = exec_time[:4]
        avg = sum(exec_time)/len(exec_time)
        times.append(avg)
    Y.append(times)
sys_legends = [f"B={bsz}" for bsz in batch_sizes]
xtick_labels = list(np.arange(1, max_tenants+1))
plot_bars(Y, sys_legends, xtick_labels, hatches, "Number of Tenants", "Average Execution Time (sec.)", f"results/{exp_name}")
#################################################################################################
#EXP4: Data Reduction experiment
#batch_sizes = [1000,2000,3000,4000,6000,8000]
#Y=[]
#y_split = get_output_size(["{logdir}/multitenant_exp/process_1_of_1_bs{bsz}" for bsz in batch_sizes])
#y_vanilla = get_output_size(["{logdir}/vanilla_run/process_1_of_1_bs{bsz}" for bsz in batch_sizes])
#Y.append(y_vanilla)
#Y.append(y_split)
#sys_legends = ["Vanilla", "Split"]
#plot_bars(Y, sys_legends, batch_sizes, hatches, "Batch Size", "Transferred Data (MBs)", f"results/batch_outputsizes")
#################################################################################################
