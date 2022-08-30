import torch
from torchvision.models import ResNet as resnet
from torchvision.models.resnet import Bottleneck, BasicBlock
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
class MyBasicBlock(BasicBlock):
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        sizes = []
        out = self.conv1(x)
        sizes.append(out.element_size() * out.nelement()/1024)
        out = self.bn1(out)
        sizes.append(out.element_size() * out.nelement()/1024)
        out = self.relu(out)
        sizes.append(out.element_size() * out.nelement()/1024)

        out = self.conv2(out)
        sizes.append(out.element_size() * out.nelement()/1024)
        out = self.bn2(out)
        sizes.append(out.element_size() * out.nelement()/1024)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        sizes.append(out.element_size() * out.nelement()/1024)

        return out, sizes

class MyBottleneck(Bottleneck):
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        sizes=[]

        out = self.conv1(x)
        sizes.append(out.element_size() * out.nelement()/1024)
        out = self.bn1(out)
        sizes.append(out.element_size() * out.nelement()/1024)
        out = self.relu(out)
        sizes.append(out.element_size() * out.nelement()/1024)

        out = self.conv2(out)
        sizes.append(out.element_size() * out.nelement()/1024)
        out = self.bn2(out)
        sizes.append(out.element_size() * out.nelement()/1024)
        out = self.relu(out)
        sizes.append(out.element_size() * out.nelement()/1024)

        out = self.conv3(out)
        sizes.append(out.element_size() * out.nelement()/1024)
        out = self.bn3(out)
        sizes.append(out.element_size() * out.nelement()/1024)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        sizes.append(out.element_size() * out.nelement()/1024)
        return out, sizes
class MyResNet(resnet):
    def __init__(self, *args, **kwargs):
        super(MyResNet, self).__init__(*args, **kwargs)
        self.all_layers = []
        remove_sequential(self, self.all_layers)
#        print("Length of all layers: ", len(self.all_layers))

    def forward(self, x:Tensor, start: int, end: int, need_time=False) -> Tensor:
      all_layers = []
      remove_sequential(self, all_layers)
      idx = 0
      res = []
#      res.append(x.element_size() * x.nelement()/1024)
      time_res = []
      names = []
      for idx in range(start, end):
          if idx >= len(all_layers):		#we avoid out of bounds
              break
          m = all_layers[idx]
          names.append(str(type(m)).split('.')[-1][:-2])
          layer_time = time()
          if isinstance(m, torch.nn.modules.linear.Linear):
              x = torch.flatten(x, 1)
          if isinstance(m, MyBasicBlock) or isinstance(m, MyBottleneck):
              x,sizes = m(x)
              # TODO CHANGE AGAIN
              #res.append(sum(sizes))
              res.extend(sizes)
          else:
              x = m(x)
              res.append(x.element_size() * x.nelement()/1024)
          time_res.append(time()-layer_time)
          if idx >= end:
              break
      if need_time:
          return x,torch.Tensor(res).cuda(), time_res #, names
      return x,torch.Tensor(res).cuda()

largs = {'resnet18':[MyBasicBlock, [2, 2, 2, 2]],
        'resnet34':[MyBasicBlock, [3, 4, 6, 3]],
	'resnet50':[MyBottleneck, [3, 4, 6, 3]],
	'resnet101':[MyBottleneck, [3, 4, 23, 3]],
	'resnet152':[MyBottleneck, [3, 8, 36, 3]],
	'resnext50_32x4d':[MyBottleneck, [3, 4, 6, 3]],
	'resnext101_32x8d':[MyBottleneck, [3, 4, 23, 3]],
	'wide_resnet50_2':[MyBottleneck, [3, 4, 6, 3]],
	'wide_resnet101_2':[MyBottleneck, [3, 4, 23, 3]]
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

#from utils import get_mem_consumption

#model = build_my_resnet('resnet18',1000)
#tot_layers=len(model.all_layers)
#for i in range(tot_layers):
#  server,client,vanilla = get_mem_consumption(model, i, tot_layers-2, 100, 1000)
#  print(f"Total GPU memory consumpton at split layer {i} is {server/1024} & {client/1024}, vanilla={vanilla/1024} GBs")
