import torch
from torchvision.models import AlexNet
from torch import Tensor
import torch.nn as nn

types = [torch.nn.modules.container.Sequential]
def remove_sequential(network, all_layers):
    for layer in network.children():
        if type(layer) in types:
            remove_sequential(layer, all_layers)
        else:
            all_layers.append(layer)

class MyAlexNet(AlexNet):

    def forward(self, x:Tensor, start: int, end: int) -> Tensor:
      idx = 0
      all_layers=[]
      remove_sequential(self, all_layers)
#      print("Input data size: {} KBs".format(x.element_size() * x.nelement()/1024))
      for idx in range(start, end):
          if idx >= len(all_layers):		#we avoid out of bounds
              break
          m = all_layers[idx]
          if isinstance(m, torch.nn.modules.linear.Linear):
              x = torch.flatten(x, 1)
          x = m(x)
#          print("Index {}, layer {}, tensor size {} KBs".format(idx, type(m), x.element_size() * x.nelement()/1024))
          if idx >= end:
              return x
      return x

def build_my_alexnet(num_classes=10):
    return MyAlexNet(num_classes=num_classes)

#model = build_my_alexnet(1000)
#a = torch.rand((2,3,224,224))
#res = model(a,0,10)
#res = model(res,10,40)
#print(res.shape)
