import numpy as np
import matplotlib
import os
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

def plot(xaxis, swift, app, total, filename):
  #Plot the GPU memory of both swift and app layers on top of each other in a bar plot and output it to {filename}.pdf
  figr = plt.figure(figsize=figsize)
  figs = []
  ind = np.arange(len(xaxis))
  fig1 = plt.bar(ind, swift, width, linewidth=1,label='Object storage',hatch="/",edgecolor='black')
  figs.append(fig1)
  fig2 = plt.bar(ind, app, width, bottom=swift, linewidth=1, color='orange', label="Application layer",hatch="\\",edgecolor='black')
  figs.append(fig2)
  plt.ylabel("GPU utilization (GBs)", fontsize=fontsize)
  plt.xlabel('Layer index', fontsize=fontsize)
  plt.xticks(ind, xaxis, fontsize=fontsize)
  plt.yticks(fontsize=fontsize)
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
