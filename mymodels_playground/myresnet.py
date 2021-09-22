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
        sizes.append(out.element_size() * out.nelement()/1024)
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
        sizes.append(out.element_size() * out.nelement()/1024)
        out = self.relu(out)
        sizes.append(out.element_size() * out.nelement()/1024)
        return out, sizes
class MyResNet(resnet):
    def __init__(self, *args, **kwargs):
        super(MyResNet, self).__init__(*args, **kwargs)
        self.all_layers = []
        remove_sequential(self, self.all_layers)
#        print("Length of all layers: ", len(self.all_layers))

    def forward(self, x:Tensor, start: int, end: int) -> Tensor:
      idx = 0
      res = []
      res.append(x.element_size() * x.nelement()/1024)
      time_res = []
      names = []
      for idx in range(start, end):
          if idx >= len(self.all_layers):		#we avoid out of bounds
              break
          m = self.all_layers[idx]
          names.append(str(type(m)).split('.')[-1][:-2])
          layer_time = time()
          if isinstance(m, torch.nn.modules.linear.Linear):
              x = torch.flatten(x, 1)
          if isinstance(m, MyBasicBlock) or isinstance(m, MyBottleneck):
              x,sizes = m(x)
              res.extend(sizes)
          else:
              x = m(x)
              res.append(x.element_size() * x.nelement()/1024)
          time_res.append(time()-layer_time)
          if idx >= end:
              break
      return x,res, time_res, names

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

def get_mem_consumption(model, split_idx, server_batch, client_batch):
    import numpy as np
    a = torch.rand((1,3,224,224))
    input_size = np.prod(np.array(a.size()))*4/ (1024*1024)*server_batch
    x,res1,_,names1 = model(a,0,split_idx)
    #x is the output, res is the sizes of intermediate outputs in KBs (Kilobytes)
    intermediate_input_size = np.prod(np.array(x.size()))*4/ (1024*1024)*client_batch
    x,res2,_,names2 = model(x,split_idx,100)
    #Calculating the required sizes
    params=[param for param in model.parameters()]
    mod_sizes = [np.prod(np.array(p.size())) for p in params]
    model_size = np.sum(mod_sizes)*4/ (1024*1024)
    before_inter_sizes = [np.prod(np.array(inter)) for inter in res1]
    after_inter_sizes = [np.prod(np.array(inter)) for inter in res2]
    #note that before split, we do only forward pass so, we do not store gradients
    #after split index, we store gradients so we expect double the storage
    before_split_size = np.sum(before_inter_sizes)/1024*server_batch - input_size
    after_split_size = np.sum(after_inter_sizes)/1024*2*client_batch
    total_server = input_size+model_size+before_split_size
    total_client = intermediate_input_size+model_size+after_split_size
#    print("Server side:")
#    print(f"input size: {input_size} MBs")
#    print(f"model size: {model_size} MBs")
#    print(f"intermediate outputs size: {before_split_size} MBs")
#    print(f"Total GPU memory: {total_server} MBs")
#    print("="*50)
#    print("Client side:")
#    print(f"input size: {intermediate_input_size} MBs")
#    print(f"model size: {model_size} MBs")
#    print(f"intermediate outputs size: {after_split_size} MBs")
#    print(f"Total GPU memory: {total_client} MBs")
#    print("="*50)
    return total_server, total_client

model = build_my_resnet('resnet18',1000)
for i in range(50):
  server,client = get_mem_consumption(model, i, 50, 256)
  print(f"Total GPU memory consumpton at split layer {i} is {server+client}")
