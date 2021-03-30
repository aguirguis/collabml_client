import torch
from torchvision.models import MNASNet
from torchvision.models.mnasnet import _InvertedResidual
from torch import Tensor
import torch.nn as nn
from time import time

types = [torch.nn.modules.container.Sequential, _InvertedResidual]
def remove_sequential(network, all_layers):
    for layer in network.children():
        if type(layer) in types:
            remove_sequential(layer, all_layers)
        else:
            all_layers.append(layer)

class MyMNASNet(MNASNet):
    def __init__(self, *args, **kwargs):
        super(MyMNASNet, self).__init__(*args, **kwargs)
        self.all_layers = []
        remove_sequential(self, self.all_layers)
        print("Length of all layers: ", len(self.all_layers))

    def forward(self, x:Tensor, start: int, end: int) -> Tensor:
      idx = 0
      print("Input data size: {} KBs".format(x.element_size() * x.nelement()/1024))
      res=[]
      time_res = []
      res.append(x.element_size() * x.nelement()/1024)
      for idx in range(start, end):
          if idx >= len(self.all_layers):		#we avoid out of bounds
              break
          m = self.all_layers[idx]
          layer_time = time()
          if isinstance(m, torch.nn.modules.linear.Linear):
              x = x.mean([2, 3])
          x = m(x)
          time_res.append(time()-layer_time)
          print("Index {}, layer {}, tensor size {} KBs".format(idx, type(m), x.element_size() * x.nelement()/1024))
          res.append(x.element_size() * x.nelement()/1024)
          if idx >= end:
              return x, res, time_res
      return x, res, time_res

def build_my_mnasnet(model, num_classes=10):
    weight = model[7:]
    weights = {"0_5":0.5, "0_75":0.75, "1_0":1.0, "1_3":1.3}
    assert weight in weights.keys()
    return MyMNASNet(weights[weight], num_classes=num_classes)

#model = build_my_mnasnet("mnasnet1_3",1000)
#a = torch.rand((1,3,224,224))
#res = model(a,0,10)
#res = model(res,10,150)
#print(res.shape)
