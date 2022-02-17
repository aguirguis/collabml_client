import torch
from torchvision.models import VGG
from torchvision.models.vgg import make_layers, cfgs
from torch import Tensor
import torch.nn as nn
from time import time

types = [torch.nn.modules.container.Sequential]
def remove_sequential(network, all_layers):
    for layer in network.children():
        if type(layer) in types:
            remove_sequential(layer, all_layers)
        else:
            all_layers.append(layer)

class MyVGG(VGG):
    def __init__(self, *args, **kwargs):
        super(MyVGG, self).__init__(*args, **kwargs)
        self.all_layers = []
        remove_sequential(self, self.all_layers)
#        print("Length of all layers: ", len(self.all_layers))

    def forward(self, x:Tensor, start: int, end: int, need_time=False) -> Tensor:
      idx = 0
      res=[]
#      res.append(x.element_size() * x.nelement()/1024)
      time_res=[]
      names=[]
      all_layers = []
      remove_sequential(self, all_layers)
      for idx in range(start, end):
          if idx >= len(all_layers):		#we avoid out of bounds
              break
          m = all_layers[idx]
          names.append(str(type(m)).split('.')[-1][:-2])
          layer_time = time()
          if isinstance(m, torch.nn.modules.linear.Linear):
              x = torch.flatten(x, 1)
          x = m(x)
          time_res.append(time()-layer_time)
          res.append(x.element_size() * x.nelement()/1024)
          if idx >= end:
              break
      if need_time:
          return x,torch.Tensor(res).cuda(), time_res #, names
      return x,torch.Tensor(res).cuda()

largs = {'vgg11':[cfgs['A'],False],
	  'vgg11_bn':[cfgs['A'],True],
	  'vgg13':[cfgs['B'],False],
	  'vgg13_bn':[cfgs['B'],True],
	  'vgg16':[cfgs['D'],False],
          'vgg16_bn':[cfgs['D'], True],
          'vgg19':[cfgs['E'], False],
          'vgg19_bn':[cfgs['E'], True]
	 }

def build_my_vgg(model, num_classes=10):
    global largs
    args=largs[model]
    return MyVGG(make_layers(*args), num_classes=num_classes)

#from utils import get_mem_consumption

#model = build_my_vgg('vgg19',num_classes=1000)
#tot_layers=len(model.all_layers)
#for i in range(tot_layers):
#  server,client,vanilla = get_mem_consumption(model, i, tot_layers-5, 100, 1000)
#  print(f"Total GPU memory consumpton at split layer {i} is {server/1024} & {client/1024}, vanilla={vanilla/1024} GBs")
