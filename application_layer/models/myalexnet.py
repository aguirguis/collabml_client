import torch
from torchvision.models import AlexNet
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

class MyAlexNet(AlexNet):
    def __init__(self, *args, **kwargs):
        super(MyAlexNet, self).__init__(*args, **kwargs)
        self.all_layers = []
        remove_sequential(self, self.all_layers)
#        print("Length of all layers: ", len(self.all_layers))

    def forward(self, x:Tensor, start: int, end: int) -> Tensor:
      idx = 0
#      res = [x.element_size() * x.nelement()/1024]		#this returns the sizes of the intermediate outputs in the network
      res = []
      time_res = []
      names=[]
      for idx in range(start, end):
          if idx >= len(self.all_layers):		#we avoid out of bounds
              break
          m = self.all_layers[idx]
          names.append(str(type(m)).split('.')[-1][:-2])
          layer_time = time()
          if isinstance(m, torch.nn.modules.linear.Linear):
              x = torch.flatten(x, 1)
          x = m(x)
          time_res.append(time()-layer_time)
          res.append(x.element_size() * x.nelement()/1024)
          if idx >= end:
              break
      return x, res, time_res, names

def build_my_alexnet(num_classes=10):
    return MyAlexNet(num_classes=num_classes)

#from utils import get_mem_consumption

#model = build_my_alexnet(1000)
#tot_layers=len(model.all_layers)
#for i in range(tot_layers):
#  server,client,vanilla = get_mem_consumption(model, i, tot_layers-5, 100, 1000)
#  print(f"Total GPU memory consumpton at split layer {i} is {server/1024} & {client/1024}, vanilla={vanilla/1024} GBs")
