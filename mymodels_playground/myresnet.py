import torch
from torchvision.models import ResNet as resnet
from torchvision.models.resnet import Bottleneck, BasicBlock
from torch import Tensor
import torch.nn as nn

types = [torch.nn.modules.container.Sequential]
def remove_sequential(network, all_layers):
    for layer in network.children():
        if type(layer) in types:
            remove_sequential(layer, all_layers)
        else:
            all_layers.append(layer)

class MyResNet(resnet):
    def __init__(self, *args, **kwargs):
        super(MyResNet, self).__init__(*args, **kwargs)
        self.all_layers = []
        remove_sequential(self, self.all_layers)
        print("Length of all layers: ", len(self.all_layers))

    def forward(self, x:Tensor, start: int, end: int) -> Tensor:
      idx = 0
      print("Input data size: {} KBs".format(x.element_size() * x.nelement()/1024))
      res = []
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

largs = {'resnet18':[BasicBlock, [2, 2, 2, 2]],
        'resnet34':[BasicBlock, [3, 4, 6, 3]],
	'resnet50':[Bottleneck, [3, 4, 6, 3]],
	'resnet101':[Bottleneck, [3, 4, 23, 3]],
	'resnet152':[Bottleneck, [3, 8, 36, 3]],
	'resnext50_32x4d':[Bottleneck, [3, 4, 6, 3]],
	'resnext101_32x8d':[Bottleneck, [3, 4, 23, 3]],
	'wide_resnet50_2':[Bottleneck, [3, 4, 6, 3]],
	'wide_resnet101_2':[Bottleneck, [3, 4, 23, 3]]
}
lkwargs = {'resnet18':{},
        'resnet34':{},
        'resnet50':{},
        'resnet101':{},
        'resnet152':{},
        'resnext50_32x4d':{'groups':32,'width_per_group':4},
        'resnext101_32x8d':{'groups':32,'width_per_group':8},
        'wide_resnet50_2':{'width_per_group':128},
        'wide_resnet101_2':{'width_per_group':128}
}
def build_my_resnet(model, num_classes=10):
    global largs, lkwargs
    args=largs[model]
    kwargs=lkwargs[model]
    return MyResNet(*args, num_classes=num_classes, **kwargs)

#model = build_my_resnet('resnet18',1000)
#a = torch.rand((1,3,224,224))
#res = model(a,0,10)
#res = model(res,10,100)
#print(res.shape)