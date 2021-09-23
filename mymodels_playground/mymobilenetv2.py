import torch
from torchvision.models import MobileNetV2
from torchvision.models.mobilenetv2 import ConvBNReLU, InvertedResidual
from torch import Tensor
import torch.nn as nn
from time import time

types = [torch.nn.modules.container.Sequential, ConvBNReLU, InvertedResidual]
def remove_sequential(network, all_layers):
    for layer in network.children():
        if type(layer) in types:
            remove_sequential(layer, all_layers)
        else:
            all_layers.append(layer)

class MyMobileNetV2(MobileNetV2):
    def __init__(self, *args, **kwargs):
        super(MyMobileNetV2, self).__init__(*args, **kwargs)
        self.all_layers = []
        remove_sequential(self, self.all_layers)
        print("Length of all layers: ", len(self.all_layers))

    def forward(self, x:Tensor, start: int, end: int) -> Tensor:
      idx = 0
#      print("Input data size: {} KBs".format(x.element_size() * x.nelement()/1024))
      res=[]
#      res.append(x.element_size() * x.nelement()/1024)
      time_res=[]
      names=[]
      for idx in range(start, end):
          if idx >= len(self.all_layers):		#we avoid out of bounds
              break
          m = self.all_layers[idx]
          names.append(str(type(m)).split('.')[-1][:-2])
          layer_time = time()
          if isinstance(m, torch.nn.modules.linear.Linear):
              x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
          x = m(x)
          time_res.append(time()-layer_time)
#          print("Index {}, layer {}, tensor size {} KBs".format(idx, type(m), x.element_size() * x.nelement()/1024))
          res.append(x.element_size() * x.nelement()/1024)
          if idx >= end:
              break
      return x,res, time_res, names

def build_my_mobilenetv2(num_classes=10):
    return MyMobileNetV2(num_classes=num_classes)

from utils import get_mem_consumption
model = build_my_mobilenetv2(1000)
tot_layers=len(model.all_layers)
for i in range(tot_layers):
  server,client,vanilla = get_mem_consumption(model, i, tot_layers-5, 100, 1000)
  print(f"Total GPU memory consumpton at split layer {i} is {server/1024} & {client/1024}, vanilla={vanilla/1024} GBs")
