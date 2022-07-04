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
projectdir = os.path.join(homedir, "collabml_client/application_layer")
font_dirs = [os.path.join(projectdir, 'experiments','./latin-modern-roman')]
font_files = mgr.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    mgr.fontManager.addfont(font_file)
#font_list = mgr.createFontList(font_files)
#mgr.fontManager.ttflist.extend(font_list)
plt.rcParams['font.family'] = 'Latin Modern Roman'
fontsize=40
figsize = (15, 12)
##Defining all the models we support
all_models = ["alexnet", 'densenet121','densenet161','densenet169','densenet201',"mobilenetv2",
		"mnasnet0_5","mnasnet0_75","mnasnet1_0","mnasnet1_3","vgg11","vgg13","vgg16","vgg19",
		"resnet18",'resnet34','resnet50','resnet101','resnet152','resnext50_32x4d',
	        'resnext101_32x8d','wide_resnet50_2','wide_resnet101_2', "inception"]

model_to_idx = {"alexnet": [0], "densenet":np.arange(1,5), "mobilenet":[5],
	"vgg":np.arange(10,14), "resnet1":np.arange(14,19), "resnet2":np.arange(19,23),
	"mnasnet":np.arange(6,10), "inception":[23]}
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
criterion = nn.CrossEntropyLoss()
for mod_class, idxs in model_to_idx.items():
#  if mod_class != 'densenet':
#    continue
  a = torch.rand((1,3,224,224))
  for ii,idx in enumerate(idxs):
    figr = plt.figure(figsize=figsize)
    figs = []
    model_str = all_models[idx]
    print("{}{}{}".format("="*20,model_str,"="*20))
    if model_str == "inception":
      a = torch.rand((2,3,299,299))
    num_classes=1000
    net = build_model(model_str, num_classes)
    res, sizes, times, names = net(a, 0,150)		#This will print some stuff
    print(names, len(names))
    target = torch.tensor((10,))
    back_time = time.time()
    loss = criterion(res, target)
    loss.backward()
    back_time = time.time()-back_time
    print("Forward iteration takes: ", np.sum(times))
    print("Backward iteration takes: ", back_time)
#    times.append(back_time)
#    names.append('Backward')
#    sizes[0] = 132			#correct the input size (before rescaling)...we know that onr ImageNet image is 132 KB on average
#    times.insert(0,0)			#basically the input layer takes no time
#    names.insert(0,'input')
    del sizes[0]
    times = np.array(times)*1000
    ind = np.arange(len(sizes))
    fig = plt.bar(ind, sizes, width, linewidth=1, hatch="/", label='Size of output data', edgecolor='black', color='blue')
    figs.append(fig)
#    ind = np.append(ind, len(ind))		#to match the backward time
#    fig3 = ax2.bar(ind+0.5*width, times, width, linewidth=1, color='orange', label=model_str+"-latency",hatch="\\",edgecolor='black',)
#    figs.append(fig3)
    figtemp, = plt.plot(ind, [132]*len(ind), color='r', linestyle="--", marker="v", ms=15)
    fig2 = plt.axhline(132, color='r', linestyle="--", label='Imagenet image size', marker="v", ms=15)
    figs.append(fig2)
#    figtemp = plt.plot(ind, [360]*len(ind), color='y', linestyle="-", marker="^", ms=15)
#    fig3 = plt.axhline(360, color='y', linestyle="-",  label='iNatura image size', marker="^", ms=15)
#    figs.append(fig3)
#    figtemp = plt.plot(ind, [1586]*len(ind), color='g', linestyle=":", marker="o", ms=15)
#    fig4 = plt.axhline(1586, color='g', linestyle=":", label='PlantLeaves image size', marker="o", ms=15)
#    figs.append(fig4)
    plt.ylabel("Output Size (KBs)", fontsize=fontsize)
    plt.xlabel('Layers', fontsize=fontsize)
    plt.xticks(ind, [name.title() for name in names], fontsize=fontsize, rotation=90)	#especially small font size because labels are big!
    plt.yticks(fontsize=fontsize)
    plt.legend(handles=figs, fontsize=fontsize, loc="upper right")
    plt.tight_layout()
    plt.savefig('{}.pdf'.format(model_str))
