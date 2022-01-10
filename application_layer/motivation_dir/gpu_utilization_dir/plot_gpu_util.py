import numpy as np
import matplotlib
import os
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

def plot(xaxis, swift, app, total, filename):
  #Plot the GPU memory of both swift and app layers on top of each other in a bar plot and output it to {filename}.pdf
  figr = plt.figure(figsize=figsize)
  figs = []
  ind = np.arange(len(xaxis))
  fig1 = plt.bar(ind, swift, width, linewidth=1,label='Before split',hatch="/",edgecolor='black', color='blue')
  figs.append(fig1)
  fig2 = plt.bar(ind, app, width, bottom=swift, linewidth=1, label="After split",hatch="\\",edgecolor='black', color='orange')
  figs.append(fig2)
  plt.ylabel("GPU Memory (GBs)", fontsize=fontsize)
  plt.xlabel('Split index', fontsize=fontsize)
  plt.xticks(ind, xaxis, fontsize=35)
  ax = plt.gca()
  tick = ax.get_xticklabels()[-1]
  tick.set_rotation(30)
  plt.yticks(fontsize=fontsize)
  plt.legend(handles=figs, fontsize=fontsize, loc="upper right")
  plt.tight_layout()
  plt.savefig('{}.pdf'.format(filename))

dirname="logFiles"
prefix='excelsheetdump_'
for filename in os.listdir(dirname)+ ["excelsheetdump_vgg11"]:
  if filename.startswith(prefix):
    model = filename[len(prefix):]
    if model == "alexnet":
        xaxis = list(np.arange(20)+1)
        swift = [1494.0, 1494.0, 1400.0, 1724.0, 1724.0, 1512.0, 1576.0, 1512.0, 1576.0, 1576.0, 1576.0, 1408.0, 1408.0, 1408.0, 1512.0, 1446.0, 1446.0, 1520.0, 1408.0, 0]
        app = [9070.0, 9070.0, 5766.0, 7172.0, 8910.0, 7934.0, 2678.0, 5084.0, 5888.0, 2500.0, 2440.0, 3174.0, 2352.0, 2490.0, 2478.0, 2452.0, 2314.0, 1353.0, 2728.0, 10904.0]
        swift, app = list(np.array(swift)/1024), list(np.array(app)/1024)
        total = list(np.array(swift)+np.array(app))
    elif model == "vgg11":
        xaxis = list(np.arange(19)+12)
        swift = [6154.0, 6154.0, 6154.0, 6154.0, 6154.0, 6154.0, 6154.0, 6154.0, 6154.0, 6154.0, 6154.0, 6154.0, 6162.0, 6162.0, 6162.0, 6162.0, 6162.0, 6162.0, 0]
        app = [25850.0, 21320.0, 21320.0, 13392.0, 13392.0, 7666.0, 10880.0, 7666.0, 3056.0, 1621.0, 2730.0, 2798.0, 3310.0, 2586.0, 3310.0, 2618.0, 2732.0, 2564.0, 13032.0]
        swift, app = list(np.array(swift)/1024), list(np.array(app)/1024)
        total = list(np.array(swift)+np.array(app))
    else:
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
    xaxis[-1] = "Status quo"
    plot(xaxis, swift, app, total, 'gpuutil_'+model)
