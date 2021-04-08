import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fontsize=35
figsize = (30, 20)
width=0.4
#used models in these experiments:
models=['resnet18', 'resnet50', 'vgg11','vgg19', 'alexnet']
ind = np.arange(len(models))
def plot(gpumem, total, filename):
  #Plot GPU memory and total time in a bar plot and output it to {filename}.pdf
  fig, ax1 = plt.subplots(figsize=figsize)
  ax2 = ax1.twinx()
  ax2.set_yticks(np.arange(0,50,step=10))
  figs = []
  fig1 = ax1.bar(ind-0.5*width, total, width, linewidth=1, label="Time",hatch="/",edgecolor='black')
  figs.append(fig1)
  fig2 = ax2.bar(ind+0.5*width, gpumem, width, linewidth=1, color='orange', label="GPU mem.",hatch="\\",edgecolor='black')
  figs.append(fig2)
  ax1.set_ylabel("Time (sec.)", fontsize=fontsize)
  ax2.set_ylabel("GPU memory (GBs)", fontsize=fontsize)
  ax1.set_xlabel('Models', fontsize=fontsize)
  ax1.tick_params(axis='y', labelsize=fontsize)
  ax2.tick_params(axis='y', labelsize=fontsize)
  ax1.tick_params(axis='x', labelsize=fontsize, rotation=90)
  plt.xticks(ind, models, fontsize=fontsize, rotation=90)
  plt.legend(handles=figs, fontsize=fontsize, loc="upper right")
  plt.tight_layout()
  plt.savefig('{}.pdf'.format(filename))

f = open('appLog','r')
lines = f.readlines()
total = []
for line in lines:
#The whole process time printing
  if line.startswith('The whole'):
    total.append(float(line.split()[-2]))
    continue
f = open('swiftLog','r')
lines = f.readlines()
gpumem = []
for line in lines:
#The whole process time printing
  if line.startswith('GPU'):
    gpumem.append(float(line.split()[-1]))
    continue
print("GPU mem: ", gpumem)
print("Total: ", total)
plot(gpumem, total, "motivation_swiftOnly")
