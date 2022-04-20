#!/usr/bin/env python3
# coding: utf-8
#This is the main script to parse the log files and plot the paper figures
import os
import numpy as np
from myparser import *
from myplotter import *
homedir = os.path.expanduser("~")
projectdir = os.path.join(homedir, "collabml_client/application_layer")
logdir = os.path.join(projectdir,"logs")
hatches = ['//','\\','x','o',"*",".", "--", "O","+"]
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1)), 'solid', '--']
#This will be a monolithic script with no functions
#It should be somehow the mirror of run_exp.py
BASELINE="Baseline"
SPLIT="HAPI"
#EXP1: models exp
exp_name="models_exp"
specific_dir = os.path.join(logdir, exp_name)
#models=['resnet18', 'resnet50', 'vgg11','vgg19', 'alexnet', 'densenet121']
models=['vit']
#logs are stored in {specific_dir}/{vanilla_or_split}_{model}_bs{batch_size}_{cpu_or_gpu}
devs=['gpu','cpu']
for dev in devs:
    Y=[]
    bsz=200
    Y.append([os.path.join(specific_dir,f"vanilla_{model}_bs{bsz}_{dev}") for model in models])
    Y.append([os.path.join(specific_dir,f"split_{model}_bs{bsz}_{dev}") for model in models])
    bsz=1000
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
    sys_legends = [f"{BASELINE}, B=200",f"{SPLIT}, B=200",f"{BASELINE}, B=1000",f"{SPLIT}, B=1000"]
    xtick_labels = [model.title() for model in models]
    colors = ["blue", "orange", "deepskyblue","darkorange"]
    print(dev)
    for i in range(0,len(Y),2):
        speedup = []
        for y1,y2 in zip(Y[i],Y[i+1]):
            if y1 != 0:
                speedup.append(y1/y2)
        if len(speedup) != 0:
            print(f"Average speedup: {sum(speedup)/len(speedup)}")
        print(f"All speedups: {speedup}")
    print("Raw values")
    for y in Y:
        print(y)
    plot_bars(Y, sys_legends, xtick_labels, hatches, "Models", "Execution Time (sec.)", f"results/{exp_name}_{dev}", text=text,colors=colors, rotation=30)
##################################################################################################
#EXP2: BW exp
exp_name="bw_exp"
specific_dir = os.path.join(logdir, exp_name)
#BW = [50*1024, 100*1024, 500*1024, 1024*1024, 2*1024*1024, 3*1024*1024, 5*1024*1024, 10*1024*1024, 15*1024*1024]
BW = [1024*1024, 12*1024*1024]
Y=[]
Y.append([os.path.join(specific_dir,f"vanilla_{bw/1024}_vit") for bw in BW])
Y.append([os.path.join(specific_dir,f"split_{bw/1024}_vit") for bw in BW])
exec_time = [get_total_exec_time(filenames) for filenames in Y]
print(f"Bandwidth speedup: {np.array(exec_time[0])/np.array(exec_time[1])}")
sys_legends = [f"{BASELINE}", f"{SPLIT}"]
xtick_labels = [int(bw/(1024*1024)) for bw in BW]
#xtick_labels[:3] = [0.05,0.1,0.5]
plot_bars(exec_time, sys_legends, xtick_labels, hatches, "Bandwidth (Gbps)", "Execution Time (sec.)", f"results/{exp_name}_exectime")
#######Another thing we want to see is the split index in each case
#split_idxs =[get_split_idx(Y[1])]
#plot_bars(split_idxs, sys_legends[1:], xtick_labels, hatches, "Bandwidth (Gbps)", "Split Index", f"results/{exp_name}_splitidx")
########THe third thing to plot is the output size in each case
#output_sizes = [get_output_size(y) for y in Y]
#plot_bars(output_sizes,sys_legends, xtick_labels, hatches, "Bandwidth (Gbps)", "Transferred Data (MBs)",f"results/{exp_name}_outputsizes")
##################################################################################################
##EXP3: Scalability with multiple tenants exp
#exp_name="multitenant_exp"
#specific_dir = os.path.join(logdir, exp_name)
#batch_sizes = [500, 1000,2000, 4000]
#max_tenants = 5
#Y = []
#for batch_size in batch_sizes:
#    times = []
#    for t in range(max_tenants):
#        num_tenants = t+1
#        filenames = [os.path.join(specific_dir,f"process_{idx+1}_of_{num_tenants}_bs{batch_size}") for idx in range(num_tenants)]
#        exec_time=get_total_exec_time(filenames)
#        if len(exec_time) > 4:
#            exec_time = exec_time[:4]
#        avg = sum(exec_time)/len(exec_time)
#        times.append(avg)
#    Y.append(times)
#sys_legends = [f"B={bsz}" for bsz in batch_sizes]
#xtick_labels = list(np.arange(1, max_tenants+1))
#plot_bars(Y, sys_legends, xtick_labels, hatches, "Number of Tenants", "Average Execution Time (sec.)", f"results/{exp_name}")
###Trial 2 for the same experiment....here, we do not do actual training so that the client cannot crash
#exp_names=["multitenant_exp_notraining"] #,"multitenant_exp_vanilla_notraining"]
##batch_sizes = [1000, 4000]
#Y=[]
#max_tenants = 11
#num_tenants = np.arange(2,max_tenants,2)
#for exp_name in exp_names:
#    specific_dir = os.path.join(logdir, exp_name)
#    times, times2 = [], []
#    for n in num_tenants:
#        filenames = [os.path.join(specific_dir,f"process_{idx+1}_of_{n}_bs1000") for idx in range(n)]
#        exec_time=get_total_exec_time(filenames)
#        exec_time = [et for et in exec_time if et != 0]		#remove zeros
#        print(f"Num of tenants: {n}, finished processes: {len(exec_time)}")
#        avg = sum(exec_time)/len(exec_time)
#        maxi = max(exec_time)
#        times.append(maxi)
#        times2.append(avg)
#    Y.append(times)
#    Y.append(times2)
#linear = [Y[0][0]*n/num_tenants[0] for n in num_tenants]
##The following two rows come from logs/swiftOnly_exp
#swiftonly_avg = [446.4932259321213, 715.1012377738953, 1521.1284693876903, 3418.3286892473698, 6968.817616629601]
#swiftonly_makespan = [449.2653920650482, 718.3016254901886, 1797.2527532577515, 3745.4075553417206, 8045.859074831009]
#Y.append(linear)
#Y.append(swiftonly_makespan)
#Y.append(swiftonly_avg)
#print("Results of multi-tenant experiment")
#for y in Y:
#    print(y)
#gains = np.array(Y[4])/np.array(Y[1])
#print(f"Average gain on average JCT of {SPLIT} compared to the trivial solution is {sum(gains)/len(gains)}")
#print(f"All gains: {gains}")
##sys_legends = [f"B={bsz}" for bsz in batch_sizes]
##sys_legends.append("Linear")
#sys_legends = [f"{SPLIT} - Makespan", f"{SPLIT} - Average" ,"Linear", "ALL_IN_COS - Makespan", "ALL_IN_COS - Average"]
#markers = ["o","x",None, "v", "^"]
#plot_lines(num_tenants, Y, sys_legends, linestyles, "Number of Tenants", "Execution Time (sec.)", f"results/{exp_names[0]}", markers)
##################################################################################################
##3a: throughput comparison: split to Vanilla with up to 6 tenants (batch size of 500)
#Y = []
#sys_legends = ["vanilla", "split"]
#max_tenants = 3 #6
#Y=[]
#for system in sys_legends:
#    specific_dir = os.path.join(logdir, "multitenant_exp" if system == "split" else "multitenant_exp_vanilla")
#    times = []
#    for t in range(max_tenants):
#        num_tenants = t+1
#        filenames = [os.path.join(specific_dir,f"process_{idx+1}_of_{num_tenants}_bs2000") for idx in range(num_tenants)]
#        exec_time=get_total_exec_time(filenames)
#        avg = sum(exec_time)/len(exec_time)
#        times.append(avg)
#    Y.append(times)
#xtick_labels = list(np.arange(1, max_tenants+1))
#sys_legends = [f"{BASELINE}", f"{SPLIT}"]
#plot_bars(Y, sys_legends, xtick_labels, hatches, "Number of Tenants", "Average Execution Time (sec.)", f"results/throughput_vanilla_split")
##################################################################################################
##EXP4: Data Reduction experiment
#batch_sizes = [1000,2000,3000,4000,6000,8000]
#Y=[]
#y_split = get_output_size([f"{logdir}/multitenant_exp/process_1_of_1_bs{bsz}" for bsz in batch_sizes])
#y_vanilla = get_output_size([f"{logdir}/vanilla_run/process_1_of_1_bs{bsz}" for bsz in batch_sizes])
#Y.append(y_vanilla)
#Y.append(y_split)
#sys_legends = [f"{BASELINE}", f"{SPLIT}"]
#print(f"Data reduction: {np.array(Y[0])/np.array(Y[1])}")
#plot_bars(Y, sys_legends, batch_sizes, hatches, "Batch Size", "Transferred Data (MBs)", f"results/batch_outputsizes")
##################################################################################################
##EXP 5: Batch adaptation experiments
##5a) compare exec time of with batch adaptation to no batch adaptation
#batch_sizes = [1000,2000,3000,4000, 6000,7000,8000]
#specific_dir = os.path.join(logdir, "batchAdatp_exp")	#no batch adaptation
#filenames = [os.path.join(specific_dir,f"process_1_of_1_bs{bsz}") for bsz in batch_sizes]
#exec_time_noadapt=get_total_exec_time(filenames)
#specific_dir = os.path.join(logdir, "withbatchAdatp_exp")   #no batch adaptation
#filenames = [os.path.join(specific_dir,f"process_1_of_1_bs{bsz}") for bsz in batch_sizes]
#exec_time_withadapt=get_total_exec_time(filenames)
#Y=[exec_time_noadapt, exec_time_withadapt]
##Add texts to our bars:
#text = []
#for y in Y:
#    t=[]
#    for yy in y:
#        s = "X" if yy==0 else ""
#        t.append(s)
#    text.append(t)
#sys_legends = ["Without BA", "With BA"]
#plot_bars(Y, sys_legends, batch_sizes, hatches, "Batch Size", "Execution Time (sec.)", f"results/batchAdapt_exectime", text)
##5b: GPU memory consumption on the server side (with and without batch adaptation)
##5c: Percentage of times the client requested batch size was too mcuh
##5d: the amount of reduction (to the batch size) the server has decided with different batch sizes
#server_metrics = {"gpu_mem":get_gpu_mem_cons, "percent_mismatch_batch":get_percent_mismatch_bs,"reduction_bs":get_reduction_bs}
#metrics_labels = {"gpu_mem": "GPU Memory (GBs)", "percent_mismatch_batch":"Batch Reduction Percentage (%)", "reduction_bs":"Average Batch Reduction(%)"}
#for k,v in server_metrics.items():
#    filenames = [os.path.join(logdir,"server_logs",f"server_noadaptation_b{bsz}") for bsz in batch_sizes]
#    metric_noadapt = v(filenames)
#    filenames = [os.path.join(logdir,"server_logs",f"server_withadaptation_b{bsz}_afterfix") for bsz in batch_sizes]
#    metric_withadapt = v(filenames)
#    print(f"{k}:{metric_withadapt}")
#    Y=[metric_noadapt, metric_withadapt]
#    if k=='gpu_mem':
#        for i in range(len(Y)):
#            Y[i] = list(np.array(Y[i])/1024)
#    #Add texts to our bars:
#    text = []
#    for y in Y:
#        t=[]
#        for yy in y:
#            s = "X" if yy==0 else ""
#            t.append(s)
#        text.append(t)
#    sys_legends = ["Without BA", "With BA"]
#    plot_bars(Y, sys_legends, batch_sizes, hatches, "Batch Size", metrics_labels[k], f"results/batchAdapt_{k}", text)
#################################################################################################
##EXP 6: GPU memory consumption in Split (server+client) compared to vanilla
##In 6a) we set the server batch to 1000 to show that the server can utilize well its GPU
##batch_sizes = [1000,2000,3000,4000, 5000, 6000,7000,8000]
#batch_sizes = [2000,4000,6000,8000,10000,12000]
#Y=[]
#dirs={"vanilla_run":"process_1_of_1_bs", "server_logs":"server_withadaptation_b","withbatchAdatp_exp":"process_1_of_1_bs"}
#for k,v in dirs.items():
#    suffix= "_afterfix" if k=="server_logs" else ""
#    filenames = [os.path.join(logdir, k, f"{v}{bsz}{suffix}") for bsz in batch_sizes]
#    Y.append(get_gpu_mem_cons(filenames))
#text = [[""]*len(y) for y in Y]
#Y[0][-1] = 0
#text[0][-1]="X"	#This is a crashing instance I know
#sys_legends = [f"{BASELINE}", f"{SPLIT} COS", f"{SPLIT} Client"]
#print("GPU memory of COS: ", Y[1])
#for i in range(len(Y)):
#    Y[i] = list(np.array(Y[i])/1024)
#plot_bars(Y, sys_legends, batch_sizes, hatches, "Batch Size", "GPU Memory (GBs)", f"results/gpumem_break_b1000", stack=[-1,-1,1], text=text)
##In 6b) we set the server batch to 200 to show that in some cases, the aggregate memory is less than the vanilla memory
#Y=[]
#dirs={"vanilla_run":"process_1_of_1_bs", "server_logs":"server_withadaptation_b","withbatchAdatp_exp":"process_1_of_1_bs"}
#for k,v in dirs.items():
#    suffix= "_afterfix_b200" if k=="server_logs" else ""
#    filenames = [os.path.join(logdir, k, f"{v}{bsz}{suffix}") for bsz in batch_sizes]
#    Y.append(get_gpu_mem_cons(filenames))
#Y[0][-1] = 0
#for i in range(len(Y)):
#    Y[i] = list(np.array(Y[i])/1024)
#sys_legends = [f"{BASELINE}", f"{SPLIT} COS", f"{SPLIT} Client"]
#plot_bars(Y, sys_legends, batch_sizes, hatches, "Batch Size", "GPU Memory (GBs)", f"results/gpumem_break_b200", stack=[-1,-1,1], text=text)
##################################################################################################
##EXP 7: Compare computation inside to outside Swift
#exp_name= "swift_inout"
#specific_dir = os.path.join(logdir, exp_name)
#models=['resnet18', 'resnet50', 'alexnet', 'densenet121']
#Y=[]
#Y.append([os.path.join(specific_dir,f"swiftin_{model}_bs2000_gpu") for model in models])
#Y.append([os.path.join(specific_dir,f"swiftout_{model}_bs2000_gpu") for model in models])
#Y = [get_total_exec_time(filenames) for filenames in Y]
#print("Inside vs. Outside Swift computation times: ")
#for y in Y:
#    print(y)
#sys_legends = ["Inside Swift", "Outside Swift"]
#xtick_labels = [model.title() for model in models]
#plot_bars(Y, sys_legends, xtick_labels, hatches, "Models", "Execution Time (sec.)", f"results/{exp_name}")
###################################################################################################
