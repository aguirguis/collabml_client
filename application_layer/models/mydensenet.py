import torch
import torchvision.models as models
from torchvision.models import DenseNet
from torchvision.models.densenet import _DenseLayer, _Transition, _DenseBlock
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from time import time

types = [torch.nn.modules.container.Sequential, _DenseLayer, _Transition]
def remove_sequential(network, all_layers):
    for layer in network.children():
        if type(layer) in types:
            remove_sequential(layer, all_layers)
        else:
            all_layers.append(layer)

class MyDenseNet(DenseNet):
    def __init__(self, *args, **kwargs):
        super(MyDenseNet, self).__init__(*args, **kwargs)
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
              x = F.relu(x, inplace=True)
              # TODO REMOVE APPEND
              res.append(x.element_size() * x.nelement()/1024.)
              x = F.adaptive_avg_pool2d(x, (1, 1))
              # TODO REMOVE APPEND
              res.append(x.element_size() * x.nelement()/1024.)
              x = torch.flatten(x, 1)
              # TODO REMOVE APPEND
              res.append(x.element_size() * x.nelement()/1024.)
          x = m(x)
          time_res.append(time()-layer_time)
          res.append(x.element_size() * x.nelement()/1024.)
          if idx >= end:
              break
      if need_time:
          return x,torch.Tensor(res).cuda(), time_res #, names
      return x,torch.Tensor(res).cuda()

largs = {'densenet121':[32, (6, 12, 24, 16), 64],
	'densenet161':[48, (6, 12, 36, 24), 96],
	'densenet169':[32, (6, 12, 32, 32), 64],
	'densenet201':[32, (6, 12, 48, 32), 64]
}

def build_my_densenet(model, num_classes=10):
    global largs
    args=largs[model]
    return MyDenseNet(*args, num_classes=num_classes)

#from utils import get_mem_consumption

#model = build_my_densenet('densenet201',1000)
#tot_layers=len(model.all_layers)
#for i in range(tot_layers):
#  server,client,vanilla = get_mem_consumption(model, i, tot_layers-10, 100, 1000)
#  print(f"Total GPU memory consumpton at split layer {i} is {server/1024} & {client/1024}, vanilla={vanilla/1024} GBs")
