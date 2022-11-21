import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as mgr
plt.rcParams['pdf.fonttype'] = 42
homedir = os.path.expanduser("~")
projectdir = os.path.join(homedir, "collabml_client/application_layer")
font_dirs = [os.path.join(projectdir, 'experiments','./latin-modern-roman')]
font_files = mgr.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    mgr.fontManager.addfont(font_file)
#font_list = mgr.createFontList(font_files)
#mgr.fontManager.ttflist.extend(font_list)
plt.rcParams['font.family'] = 'Latin Modern Roman'
fontsize=40
figsize = (15, 10)
width=0.4
colors = ["blue", "orange", "darkblue","darkorange"]
#used models in these experiments:
models=['resnet18', 'resnet50', 'vgg11','vgg19', 'alexnet', 'densenet121']
print(models)
#models=['resnet18', 'resnet50', 'vgg11', 'vgg19', 'alexnet']
ind = np.arange(len(models))
def plot(comp, comm, total, filename):
  #Plot computation and communication time in a bar plot and output it to {filename}.pdf
  figr = plt.figure(figsize=figsize)
  figs = []
  fig1 = plt.bar(ind, comp, width, linewidth=1, label="Computation",hatch="/",edgecolor='black', color=colors[0])
  figs.append(fig1)
  fig2 = plt.bar(ind, comm, width, bottom=comp, linewidth=1, color=colors[1], label="Communication",hatch="\\",edgecolor='black')
  figs.append(fig2)
  #put 'X' on crashing bars...
  text = []
  for t in total:
      text.append("X" if t==0 else "")
  for s,rect in zip(text,fig2):
      height = rect.get_height()
      plt.text(rect.get_x() + rect.get_width()/2.0, height, s, ha='center', va='bottom', weight='bold', color="red", fontsize=30)
  plt.ylabel("Execution Time (sec.)", fontsize=fontsize)
  plt.xlabel('Models', fontsize=fontsize)
  plt.xticks(ind, [model.title() if model.title() != 'Vit' else 'Transformer' for model in models], fontsize=fontsize, rotation=30)
  plt.yticks(fontsize=fontsize)
  plt.legend(handles=figs, fontsize=fontsize, loc="upper right")
  plt.tight_layout()
  plt.savefig('{}.pdf'.format(filename))
  plt.gcf().clear()

#bws=['UNLIMITED', 150, 100]
bws=[150]
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
        print(line)
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
    #del comp[2], comm[2], total[2]
    #del comp[-1], comm[-1], total[-1]
    total = [total[i] if total[i] > 50 else 0 for i in range(len(total))]
    comp = [comp[i] if comp[i] > 50 and total[i] != 0 else 0 for i in range(len(comp))]
    comm = [comm[i] if comm[i] > 50 and total[i] != 0 else 0 for i in range(len(comm))]
    plot(comp, comm, total, "motivation_bw{}_{}".format(bw, "GPU" if ext == "" else ext))
    print("Computation: ", comp)
    print("Communication: ", comm)
    print("Total: ", total)
