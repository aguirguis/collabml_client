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

devs=['gpu']
models = ['vit']
bss = [200, 1000, 4000, 8000]
oss = [200, 1000]
for dev in devs:
  Y=[]
  Y.append([os.path.join(logdir,f"vanilla_{model}_bs{200}_os{200}_{dev}") for model in models])
  Y.append([os.path.join(logdir,f"vanilla_{model}_bs{1000}_os{200}_{dev}") for model in models])
  Y.append([os.path.join(logdir,f"split_{model}_bs{4000}_os{1000}_{dev}") for model in models])
  Y.append([os.path.join(logdir,f"split_{model}_bs{8000}_os{1000}_{dev}") for model in models])
  param_os_bs = [(1000, 4000), (1000, 8000)]
  #for o_s in oss:
  #  for b_s in bss:
  #      Y.append([os.path.join(logdir,f"split_{model}_bs{b_s}_os{o_s}_{dev}") for model in models])
  #      param_os_bs.append((o_s,b_s))

  times=[]
  for filenames in Y:
    try:
      time = get_total_exec_time(filenames)
      times.append(time)
    except:
      pass
  Y = times
  print(Y)
  #Add texts to our bars:
  text = []
  for y in Y:
      t=[]
      for yy in y:
          s = "X" if yy==0 else ""
          t.append(s)
      text.append(t)

  #assert len(Y) <= len(sys_legends) and len(Y[0]) == len(xtick_labels)
  sys_legends = [f"{BASELINE}, BS=200, OS=200", f"{BASELINE}, BS=1000, OS=200"]
  for o_s, b_s in param_os_bs:
    sys_legends.append(f"{SPLIT}, BS={b_s}, OS={o_s}")
  xtick_labels = [model.title() for model in models]

  #sys_legends = [f"{BASELINE}, B=2000",f"{SPLIT}, B=2000",f"{BASELINE}, B=8000",f"{SPLIT}, B=8000"]
  #xtick_labels = [f"{os}, {bs}" for os_bs in param_os_bs]
  #colors = ["blue", "orange", "deepskyblue","darkorange"]
  plot_bars(Y, sys_legends, xtick_labels, hatches, "VIT increasing Batch Size", "Execution Time (sec.)", f"results/vit", text=text, rotation=30)
