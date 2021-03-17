import torch
from torchvision.models.inception import Inception3, InceptionAux
from torch import Tensor
import torch.nn as nn

types = [torch.nn.modules.container.Sequential]
def remove_sequential(network, all_layers):
    for layer in network.children():
        if type(layer) in types:
            remove_sequential(layer, all_layers)
        else:
            all_layers.append(layer)

class MyInception(Inception3):

    def forward(self, x:Tensor, start: int, end: int) -> Tensor:
      idx = 0
      all_layers=[]
      remove_sequential(self, all_layers)
#      print("Input data size: {} KBs".format(x.element_size() * x.nelement()/1024))
      aux = None
      for idx in range(start, end):
          if idx >= len(all_layers):		#we avoid out of bounds
              break
          m = all_layers[idx]
          if isinstance(m, InceptionAux):
              aux = m(x)
              continue
          if isinstance(m, torch.nn.modules.linear.Linear):
              x = torch.flatten(x, 1)
          x = m(x)
#          print("Index {}, layer {}, tensor size {} KBs".format(idx, type(m), x.element_size() * x.nelement()/1024))
          if idx >= end:
              return x
      return x			#TODO: check if we need to return aux also with this

def build_my_inception(num_classes=10):
    return MyInception(num_classes=num_classes)

#model = build_my_inception(1000)
#a = torch.rand((2,3,299,299))
#res = model(a,0,10)
#res = model(res,10,40)
#print(res.shape)
