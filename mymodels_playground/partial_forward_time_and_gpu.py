import os
import torch
import torch.nn as nn
from myresnet import build_my_resnet
from myvgg import build_my_vgg
from mydensenet import build_my_densenet
from myalexnet import build_my_alexnet
from myinception import build_my_inception
from mymnasnet import build_my_mnasnet
from mymobilenetv2 import build_my_mobilenetv2
import time
###For plotting
import numpy as np
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

def build_model(model_str, num_classes):
  if model_str.startswith('alex'):
    model = build_my_alexnet(num_classes)
  elif model_str.startswith('res') or model_str.startswith('wide'):
    model = build_my_resnet(model_str, num_classes)
  elif model_str.startswith('vgg'):
    model = build_my_vgg(model_str, num_classes)
  elif model_str.startswith('dense'):
    model = build_my_densenet(model_str, num_classes)
  elif model_str.startswith('mobile'):
    model = build_my_mobilenetv2(num_classes)
  elif model_str.startswith('mnas'):
    model = build_my_mnasnet(model_str, num_classes)
  return model

criterion = nn.CrossEntropyLoss()
num_classes= 500
def partial_forward(net, split_idx, batch_size, device):
  #helper function to measure the time it takes to do forward of the last few layers
  #TODO: backward should be limited only to the last few layers (not to the input layer)
  input = torch.rand((batch_size,3,224,224))
  if device == "cuda":		#uncommenting this line makes computation very slow!
      input, net = input.cuda(), net.cuda()
  res, _, _, _ = net(input, 0,split_idx)              #This will print some stuff
  res, net = res.cpu(), net.cpu()
  target = torch.randint(1, num_classes, (batch_size,))
  ######Start the benchmark from here
  torch.cuda.empty_cache()
  torch.cuda.reset_max_memory_allocated(0)
  torch.cuda.reset_max_memory_allocated(1)
  gpu_usage_init=(torch.cuda.max_memory_allocated(0)+torch.cuda.max_memory_allocated(1))/(1024*1024*1024)
  start_time = time.time()
  res, target, net = res.to(device), target.to(device), net.to(device)
  res, _, _, _ = net(res, split_idx, 150)
  loss = criterion(res, target)
  loss.backward()
  total_time = time.time()-start_time
  gpu_usage=(torch.cuda.max_memory_allocated(0)+torch.cuda.max_memory_allocated(1))/(1024*1024*1024) - gpu_usage_init
  print("split_idx: {} device: {}, gpu_usage: {}".format(split_idx, device, gpu_usage))
  #cleaning
  del input, target, res, net
  torch.cuda.empty_cache()
  return total_time, gpu_usage

#Dict of model to split indexes
models_dict={'alexnet': np.arange(1,22), #[16,17,18,19,20,21],
	    'resnet18': np.arange(1,15), #[9,10,11,12,13,14],
	    'vgg11': np.arange(1,30), #[24,25,26,27,28,29],
	    'densenet121': np.arange(1,23) #[17,18,19,20,21,22]
}
batch_sizes=[50,100,200] #only useful for the GPU mem. plot
width=0.4
for model, split_idxs in models_dict.items():
  res_dict={}
  gpu_mems=[]
  for batch_size in batch_sizes:
    if batch_size == 200:
      devices = ['cuda'] #, 'cpu']		#we don't need the CPU....the time values were already stored hard-coded in the other file
    else:
      devices = ['cuda']
    for device in devices:
      times = []
      for split_idx in split_idxs:
        net = build_model(model, num_classes)
        total_time, gpu_usage = partial_forward(net, split_idx, batch_size, device)
        times.append(total_time)
        if device == 'cuda':		#Do this only while using GPU
          gpu_mems.append(gpu_usage)
      res_dict[device] = times
    print("Computation times of {} on {} with batch size {}: ".format(model, device, batch_size), times)
  ##Plotting results
  fig = plt.figure(figsize=figsize)
  figs = []
  ind = np.arange(len(split_idxs))
  fig = plt.bar(ind-0.5*width, res_dict['cpu'], width, linewidth=1, label="CPU",hatch="/",edgecolor='black')
  figs.append(fig)
  fig = plt.bar(ind+0.5*width, res_dict['cuda'], width, linewidth=1, label="GPU",hatch="\\",edgecolor='black')
  figs.append(fig)
  plt.ylabel("Time (sec.)", fontsize=fontsize)
  plt.xlabel('Layer index', fontsize=fontsize)
  plt.yticks(fontsize=fontsize)
  plt.xticks(ind, split_idxs, fontsize=fontsize)
  plt.legend(handles=figs, fontsize=fontsize, loc="upper right")
  plt.tight_layout()
  plt.savefig('observation_all_layers_{}.pdf'.format(model))
  ##Plotting GPU memory usage
  plt.gcf().clear()
  ind = np.arange(len(split_idxs))
  fig = plt.figure(figsize=figsize)
  figs = []
  fig = plt.bar(ind - width, gpu_mems[:len(ind)], width, linewidth=1, label="Batch = {}".format(batch_sizes[0]),edgecolor='black')
  figs.append(fig)
  fig2 = plt.bar(ind, gpu_mems[len(ind):2*len(ind)], width, linewidth=1, label="Batch = {}".format(batch_sizes[1]),edgecolor='black')
  figs.append(fig2)
  fig3 = plt.bar(ind + width, gpu_mems[2*len(ind):], width, linewidth=1, label="Batch = {}".format(batch_sizes[2]),edgecolor='black')
  figs.append(fig3)
  plt.ylabel("GPU memory (GBs)", fontsize=fontsize)
  plt.xlabel('Start layer index', fontsize=fontsize)
  plt.yticks(fontsize=fontsize)
  plt.xticks(ind, split_idxs, fontsize=fontsize)
  plt.legend(handles=figs, fontsize=fontsize, loc="upper right")
  plt.tight_layout()
  plt.savefig('observation_gpu_mem_per_layer_{}.pdf'.format(model))
