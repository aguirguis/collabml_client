import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fontsize=35
figsize = (30, 20)
width=0.4

#used models in these experiments:
models=['resnet18', 'resnet50', 'resnet152', 'vgg11', 'vgg19', 'alexnet', 'densenet121']
ind = np.arange(len(models))
def plot(comp, comm, total, filename):
  #Plot computation and communication time in a bar plot and output it to {filename}.pdf
  fig, ax1 = plt.subplots(figsize=figsize)
  figs = []
#  ax2.set_yticks()
  fig1 = ax1.bar(ind, comp, width, linewidth=1, label="Computation",hatch="/",edgecolor='black')
  figs.append(fig1)
  fig2 = ax1.bar(ind, np.array(total)-np.array(comp), width, bottom=comp, linewidth=1, color='orange', label="Communication",hatch="\\",edgecolor='black')
  figs.append(fig2)
  ax1.set_ylabel("One-epoch latency (sec.)", fontsize=fontsize)
#  ax2.set_ylabel("Time to process a layer (ms)", fontsize=fontsize)
  ax1.set_xlabel('Models', fontsize=fontsize)
  ax1.tick_params(axis='y', labelsize=fontsize)
#  ax2.tick_params(axis='y', labelsize=fontsize)
  ax1.tick_params(axis='x', labelsize=fontsize, rotation=90)
  plt.xticks(ind, models, fontsize=fontsize, rotation=90)
  plt.legend(handles=figs, fontsize=fontsize, loc="upper right")
  plt.tight_layout()
  plt.savefig('{}.pdf'.format(filename))

bws=['UNLIMITED', 150, 100]
for bw in bws:				#Bandwidth
  for ext in ["", "CPU"]:		#GPU or CPU
    filename='parallelBaselineMotivation_bw{}{}'.format(bw, ext)
    print("Processing file: ", filename)
    f = open(filename,'r')
    lines = f.readlines()
    comm = []
    total = []
    comp = []
    cur_comm=0
    cur_comp=0
    cur_total=0
    for line in lines:
    #Communication time
      if line.startswith('Streaming imagenet'):
        cur_comm+=float(line.split()[-2])
        continue
      if line.startswith('One training iteration'):
        cur_comp+=float(line.split()[-2])
        continue
    #The whole process time printing
      if line.startswith('The whole'):
        cur_total+=float(line.split()[-2])
        continue
    #The time of receiving results from Swift printing
      if line.startswith('Namespace'):
        comm.append(cur_comm)
        comp.append(cur_comp)
        total.append(cur_total)
        cur_comm=0
        cur_comp=0
        cur_total=0
    comp.append(cur_comp)
    comm.append(cur_comm)
    total.append(cur_total)
    del comp[0], comm[0], total[0]		#remove the extra entry in the beginning
    plot(comp, comm, total, "motivation_bw{}_{}".format(bw, "GPU" if ext == "" else ext))
    print("Computation: ", comp)
    print("Communication: ", comm)
    print("Total: ", total)
