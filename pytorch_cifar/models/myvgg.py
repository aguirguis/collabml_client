import torch
from torchvision.models import VGG
from torchvision.models.vgg import make_layers, cfgs
from torch import Tensor
import torch.nn as nn

types = [torch.nn.modules.container.Sequential]
def remove_sequential(network, all_layers):
    for layer in network.children():
        if type(layer) in types:
            remove_sequential(layer, all_layers)
        else:
            all_layers.append(layer)

class MyVGG(VGG):
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

args = {'vgg11':[cfgs['A'],False],
	  'vgg11_bn':[cfgs['A'],True],
	  'vgg13':[cfgs['B'],False],
	  'vgg13_bn':[cfgs['B'],True],
	  'vgg16':[cfgs['D'],False],
          'vgg16_bn':[cfgs['D'], True],
          'vgg19':[cfgs['E'], False],
          'vgg19_bn':[cfgs['E'], True]
	 }

def build_my_vgg(model, num_classes=10):
    global args
    args=args[model]
    return MyVGG(make_layers(*args), num_classes=num_classes)

#model = build_my_vgg('vgg19',num_classes=1000)
#a = torch.rand((2,3,224,224))
#res = model(a,0,10)
#res = model(res,10,40)
#print(res.shape)
