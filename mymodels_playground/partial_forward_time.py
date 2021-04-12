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
  input, net = input.cuda(), net.cuda()
  res, _, _, _ = net(input, 0,split_idx)              #This will print some stuff
  res, net = res.cpu(), net.cpu()
  target = torch.randint(1, num_classes, (batch_size,))
  ######Start the benchmark from here
  start_time = time.time()
  res, target, net = res.to(device), target.to(device), net.to(device)
  res, _, _, _ = net(res, split_idx, 150)
  loss = criterion(res, target)
  loss.backward()
  total_time = time.time()-start_time
  #cleaning
  del input, target, res, net
  torch.cuda.empty_cache()
  return total_time

#Dict of model to split indexes
models_dict={'alexnet': np.arange(1,22), #[16,17,18,19,20,21],
	    'resnet18': np.arange(1,15), #[9,10,11,12,13,14],
	    'vgg11': np.arange(1,30), #[24,25,26,27,28,29],
	    'densenet121': np.arange(1,23) #[17,18,19,20,21,22]
}
devices=['cuda', 'cpu']
batch_size=200
fontsize=35
figsize = (30, 20)
width=0.4
for model, split_idxs in models_dict.items():
  res_dict={}
  for device in devices:
    times = []
    for split_idx in split_idxs:
      net = build_model(model, num_classes)
      total_time = partial_forward(net, split_idx, batch_size, device)
      times.append(total_time)
    res_dict[device] = times
    print("Computation times of {} on {}: ".format(model, device), times)
  ##Plotting results
  fig, ax1 = plt.subplots(figsize=figsize)
  figs = []
  ind = np.arange(len(split_idxs))
  fig = ax1.bar(ind-0.5*width, res_dict['cpu'], width, linewidth=1, label="CPU",hatch="/",edgecolor='black')
  figs.append(fig)
  fig = ax1.bar(ind+0.5*width, res_dict['cuda'], width, linewidth=1, label="GPU",hatch="\\",edgecolor='black')
  figs.append(fig)
  ax1.set_ylabel("Time (sec.)", fontsize=fontsize)
  ax1.set_xlabel('Layer index', fontsize=fontsize)
  ax1.tick_params(axis='y', labelsize=fontsize)
  ax1.tick_params(axis='x', labelsize=fontsize)
  plt.xticks(ind, split_idxs, fontsize=fontsize)
  plt.legend(handles=figs, fontsize=fontsize, loc="upper right")
  plt.tight_layout()
  plt.savefig('observation_all_layers_{}.pdf'.format(model))
