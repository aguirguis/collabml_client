import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fontsize=80
figsize = (30, 20)
width=0.4
def plot(xaxis, swift, app, total, filename):
  #Plot the GPU memory of both swift and app layers on top of each other in a bar plot and output it to {filename}.pdf
  fig, ax1 = plt.subplots(figsize=figsize)
  figs = []
  ind = np.arange(len(xaxis))
  fig1 = ax1.bar(ind, swift, width, linewidth=1,label='Object storage',hatch="/",edgecolor='black')
  figs.append(fig1)
  fig2 = ax1.bar(ind, app, width, bottom=swift, linewidth=1, color='orange', label="Application layer",hatch="\\",edgecolor='black')
  figs.append(fig2)
  ax1.set_ylabel("GPU utilization (GBs)", fontsize=fontsize)
  ax1.set_xlabel('Layer index', fontsize=fontsize)
  ax1.tick_params(axis='y', labelsize=fontsize)
  ax1.tick_params(axis='x', labelsize=fontsize)
  plt.xticks(ind, xaxis, fontsize=fontsize)
  plt.legend(handles=figs, fontsize=fontsize, loc="upper right")
  plt.tight_layout()
  plt.savefig('{}.pdf'.format(filename))

dirname="logFiles"
prefix='excelsheetdump_'
for filename in os.listdir(dirname):
  if filename.startswith(prefix):
    model = filename[len(prefix):]
    f = open(os.path.join(dirname, filename),'r')
    xaxis, swift, app, total = [],[],[],[]
    for line in f.readlines():
      if line.startswith('split'):		#first line in file, IGNORE
        continue
      contents =line.split()
      xaxis.append(contents[0])
      swift.append(float(contents[1]))
      app.append(float(contents[2]))
      total.append(float(contents[3]))
    plot(xaxis, swift, app, total, 'gpuutil_'+model)
