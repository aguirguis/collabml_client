import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as mgr
plt.rcParams['pdf.fonttype'] = 42
homedir = os.path.expanduser("~")
projectdir = os.path.join(homedir, "swift_playground/application_layer")
font_dirs = [os.path.join(projectdir, 'experiments','./latin-modern-roman')]
font_files = mgr.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    mgr.fontManager.addfont(font_file)
#font_list = mgr.createFontList(font_files)
#mgr.fontManager.ttflist.extend(font_list)
plt.rcParams['font.family'] = 'Latin Modern Roman'
fontsize=40
figsize = (15, 8)
width=0.4

#used models in these experiments:
models=['resnet18', 'resnet50', 'resnet152', 'vgg11', 'vgg19', 'alexnet', 'densenet121']
ind = np.arange(len(models))
def plot(comp, comm, total, filename):
  #Plot computation and communication time in a bar plot and output it to {filename}.pdf
  figr = plt.figure(figsize=figsize)
  figs = []
  fig1 = plt.bar(ind, comp, width, linewidth=1, label="Computation",hatch="/",edgecolor='black')
  figs.append(fig1)
  fig2 = plt.bar(ind, np.array(total)-np.array(comp), width, bottom=comp, linewidth=1, color='orange', label="Communication",hatch="\\",edgecolor='black')
  figs.append(fig2)
  plt.ylabel("One-epoch latency (sec.)", fontsize=fontsize)
  plt.xlabel('Models', fontsize=fontsize)
  plt.xticks(ind, [model.title() for model in models], fontsize=fontsize)
  plt.yticks(fontsize=fontsize)
  plt.legend(handles=figs, fontsize=fontsize, loc="upper right")
  plt.tight_layout()
  plt.savefig('{}.pdf'.format(filename))
  plt.gcf().clear()

bws=['UNLIMITED', 150, 100]
for bw in bws:				#Bandwidth
  for ext in ["", "CPU"]:		#GPU or CPU
    filename='logFiles/parallelBaselineMotivation_bw{}{}'.format(bw, ext)
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
