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
hatches = ['//','\\','x','o',"*",".", "--", "O","+"]
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1)), 'solid', '--']
#This will be a monolithic script with no functions
#It should be somehow the mirror of run_exp.py
#EXP1: models exp
exp_name="models_exp"
specific_dir = os.path.join(logdir, exp_name)
models=['resnet18', 'resnet50', 'vgg11','vgg19', 'alexnet', 'densenet121']
#logs are stored in {specific_dir}/{vanilla_or_split}_{model}_bs{batch_size}_{cpu_or_gpu}
devs=['gpu','cpu']
for dev in devs:
    Y=[]
    bsz=2000
    Y.append([os.path.join(specific_dir,f"vanilla_{model}_bs{bsz}_{dev}") for model in models])
    Y.append([os.path.join(specific_dir,f"split_{model}_bs{bsz}_{dev}") for model in models])
    bsz=8000
    Y.append([os.path.join(specific_dir,f"vanilla_{model}_bs{bsz}_{dev}") for model in models])
    Y.append([os.path.join(specific_dir,f"split_{model}_bs{bsz}_{dev}") for model in models])
    Y = [get_total_exec_time(filenames) for filenames in Y]
    #Add texts to our bars:
    text = []
    for y in Y:
        t=[]
        for yy in y:
            s = "X" if yy==0 else ""
            t.append(s)
        text.append(t)
    sys_legends = ["Vanilla - B=2000","Split - B=2000","Vanilla - B=8000","Split - B=8000"]
    xtick_labels = [model.title() for model in models]
    plot_bars(Y, sys_legends, xtick_labels, hatches, "Models", "Execution Time (sec.)", f"results/{exp_name}_{dev}", text)
##################################################################################################
#EXP2: BW exp
exp_name="bw_exp"
specific_dir = os.path.join(logdir, exp_name)
BW = [50*1024, 100*1024, 500*1024, 1024*1024, 2*1024*1024, 3*1024*1024, 5*1024*1024, 10*1024*1024, 15*1024*1024]
Y=[]
Y.append([os.path.join(specific_dir,f"vanilla_{bw/1024}") for bw in BW])
Y.append([os.path.join(specific_dir,f"split_{bw/1024}") for bw in BW])
exec_time = [get_total_exec_time(filenames) for filenames in Y]
sys_legends = ["Vanilla", "Split"]
xtick_labels = [int(bw/1024) for bw in BW]
plot_bars(exec_time, sys_legends, xtick_labels, hatches, "Bandwidth (Mbps)", "Execution Time (sec.)", f"results/{exp_name}_exectime")
#######Another thing we want to see is the split index in each case
split_idxs =[get_split_idx(Y[1])]
plot_bars(split_idxs, sys_legends[1:], xtick_labels, hatches, "Bandwidth (Mbps)", "Split Index", f"results/{exp_name}_splitidx")
#######THe third thing to plot is the output size in each case
output_sizes = [get_output_size(y) for y in Y]
plot_bars(output_sizes,sys_legends, xtick_labels, hatches, "Bandwidth (Mbps)", "Transferred Data (MBs)",f"results/{exp_name}_outputsizes")
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
batch_sizes = [1000,2000,3000,4000,6000,8000]
Y=[]
y_split = get_output_size([f"{logdir}/multitenant_exp/process_1_of_1_bs{bsz}" for bsz in batch_sizes])
y_vanilla = get_output_size([f"{logdir}/vanilla_run/process_1_of_1_bs{bsz}" for bsz in batch_sizes])
Y.append(y_vanilla)
Y.append(y_split)
sys_legends = ["Vanilla", "Split"]
plot_bars(Y, sys_legends, batch_sizes, hatches, "Batch Size", "Transferred Data (MBs)", f"results/batch_outputsizes")
#################################################################################################
#EXP 5: Batch adaptation experiments
#5a) compare exec time of with batch adaptation to no batch adaptation
batch_sizes = [1000,2000,3000,4000, 6000,7000,8000]
specific_dir = os.path.join(logdir, "batchAdatp_exp")	#no batch adaptation
filenames = [os.path.join(specific_dir,f"process_1_of_1_bs{bsz}") for bsz in batch_sizes]
exec_time_noadapt=get_total_exec_time(filenames)
specific_dir = os.path.join(logdir, "withbatchAdatp_exp")   #no batch adaptation
filenames = [os.path.join(specific_dir,f"process_1_of_1_bs{bsz}") for bsz in batch_sizes]
exec_time_withadapt=get_total_exec_time(filenames)
Y=[exec_time_noadapt, exec_time_withadapt]
#Add texts to our bars:
text = []
for y in Y:
    t=[]
    for yy in y:
        s = "X" if yy==0 else ""
        t.append(s)
    text.append(t)
sys_legends = ["With Batch Adaptation", "Without Batch Adaptation"]
plot_bars(Y, sys_legends, batch_sizes, hatches, "Batch Size", "Execution Time (sec.)", f"results/batchAdapt_exectime", text)
#5b: GPU memory consumption on the server side (with and without batch adaptation)
#5c: Percentage of times the client requested batch size was too mcuh
#5d: the amount of reduction (to the batch suze) the server has decided with different batch sizes
server_metrics = {"gpu_mem":get_gpu_mem_cons, "percent_mismatch_batch":get_percent_mismatch_bs,"reduction_bs":get_reduction_bs}
metrics_labels = {"gpu_mem":"Max. GPU Mem. Consumption (MBs)", "percent_mismatch_batch":"Batch Reduction Percentage (%)", "reduction_bs":"Average Batch Reduction(%)"}
for k,v in server_metrics.items():
    filenames = [os.path.join(logdir,"server_logs",f"server_noadaptation_b{bsz}") for bsz in batch_sizes]
    metric_noadapt = v(filenames)
    filenames = [os.path.join(logdir,"server_logs",f"server_withadaptation_b{bsz}") for bsz in batch_sizes]
    metric_withadapt = v(filenames)
    Y=[metric_noadapt, metric_withadapt]
    #Add texts to our bars:
    text = []
    for y in Y:
        t=[]
        for yy in y:
            s = "X" if yy==0 else ""
            t.append(s)
        text.append(t)
    sys_legends = ["With Batch Adaptation", "Without Batch Adaptation"]
    plot_bars(Y, sys_legends, batch_sizes, hatches, "Batch Size", metrics_labels[k], f"results/batchAdapt_{k}", text)
#################################################################################################
#EXP 6: GPU memory consumption in Split (server+client) compared to vanilla
batch_sizes = [1000,2000,3000,4000, 5000, 6000,7000,8000]
Y=[]
dirs={"vanilla_run":"process_1_of_1_bs", "server_logs":"server_withadaptation_b","withbatchAdatp_exp":"process_1_of_1_bs"}
for k,v in dirs.items():
    filenames = [os.path.join(logdir, k, f"{v}{bsz}") for bsz in batch_sizes]
    Y.append(get_gpu_mem_cons(filenames))
sys_legends = ["Vanilla", "Split - Server", "Split Client"]
plot_bars(Y, sys_legends, batch_sizes, hatches, "Batch Size", "GPU Mem. Consumption (MBs)", f"results/gpumem_break", stack=[-1,-1,1])
#################################################################################################
