import torch
from myresnet import build_my_resnet
from myvgg import build_my_vgg
from mydensenet import build_my_densenet
from myalexnet import build_my_alexnet
from myinception import build_my_inception
from mymnasnet import build_my_mnasnet
from mymobilenetv2 import build_my_mobilenetv2
###For plotting
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
#import matplotlib.font_manager as mgr
fontsize=35
figsize = (30, 20)
##Defining all the models we support
all_models = ["alexnet", 'densenet121','densenet161','densenet169','densenet201',"mobilenetv2",
		"mnasnet0_5","mnasnet0_75","mnasnet1_0","mnasnet1_3","vgg11","vgg13","vgg16","vgg19",
		"resnet18",'resnet34','resnet50','resnet101','resnet152','resnext50_32x4d',
	        'resnext101_32x8d','wide_resnet50_2','wide_resnet101_2', "inception"]

model_to_idx = {"alexnet": [0], "densenet":np.arange(1,5), "mobilenet":[5], 
		"mnasnet":np.arange(6,10), "vgg":np.arange(10,14), "resnet1":np.arange(14,19), "resnet2":np.arange(19,23), "inception":[23]}
def build_model(model_str, num_classes):
  if model_str.startswith('alex'):
    model = build_my_alexnet(num_classes)
  elif model_str.startswith('inception'):
    model = build_my_inception(num_classes)
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

width=0.4
for mod_class, idxs in model_to_idx.items():
  a = torch.rand((1,3,224,224))
  for ii,idx in enumerate(idxs):
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    figs = []
    model_str = all_models[idx]
    print("{}{}{}".format("="*20,model_str,"="*20))
    if model_str == "inception":
      a = torch.rand((2,3,299,299))
    net = build_model(model_str, 1000)
    _, sizes, times = net(a, 0,150)		#This will print some stuff
    sizes[0] = 132			#correct the input size (before rescaling)...we know that onr ImageNet image is 132 KB on average
    times.insert(0,0)			#basically the input layer takes no time
    times = np.array(times)*1000
    ind = np.arange(len(sizes))
    ax2.set_yticks(np.arange(0,np.max(times),step=np.max(times)/10))
    fig = ax1.bar(ind-0.5*width, sizes, width, linewidth=1, label=model_str+"-data-size",hatch="/",edgecolor='black',)
    figs.append(fig)
    fig3 = ax2.bar(ind+0.5*width, times, width, linewidth=1, color='orange', label=model_str+"-latency",hatch="\\",edgecolor='black',)
    figs.append(fig3)
    fig2 = ax1.axhline(132, color='r', linestyle="--", label='input size')
    figs.append(fig2)
    ax1.set_ylabel("Output size per image (KBs)", fontsize=fontsize)
    ax2.set_ylabel("Time to process a layer (ms)", fontsize=fontsize)
    ax1.set_xlabel('Layer index', fontsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize)
    plt.legend(handles=figs, fontsize=fontsize, loc="upper right")
    plt.tight_layout()
    plt.savefig('{}.pdf'.format(model_str))
