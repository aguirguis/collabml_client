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
    def __init__(self, *args, **kwargs):
        super(MyVGG, self).__init__(*args, **kwargs)
        self.all_layers = []
        remove_sequential(self, self.all_layers)
        print("Length of all layers: ", len(self.all_layers))

    def forward(self, x:Tensor, start: int, end: int) -> Tensor:
      idx = 0
      print("Input data size: {} KBs".format(x.element_size() * x.nelement()/1024))
      res=[]
      res.append(x.element_size() * x.nelement()/1024)
      for idx in range(start, end):
          if idx >= len(self.all_layers):		#we avoid out of bounds
              break
          m = self.all_layers[idx]
          if isinstance(m, torch.nn.modules.linear.Linear):
              x = torch.flatten(x, 1)
          x = m(x)
          print("Index {}, layer {}, tensor size {} KBs".format(idx, type(m), x.element_size() * x.nelement()/1024))
          res.append(x.element_size() * x.nelement()/1024)
          if idx >= end:
              return x,res
      return x,res

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

#model = build_my_vgg('vgg19',num_classes=1000)
#a = torch.rand((1,3,224,224))
#res = model(a,0,10)
#res = model(res,10,100)
#print(res.shape)
