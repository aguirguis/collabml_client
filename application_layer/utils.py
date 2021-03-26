'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
from torchvision.models.densenet import _DenseLayer, _Transition		#only for freeze_sequential of densenet
import psutil
import numpy as np
import torch
import torchvision
from io import BytesIO
from PIL import Image
import pickle
from swiftclient.service import SwiftService, SwiftPostObject

try:
    from application_layer.models import *
    from application_layer.dataset_utils import *
    from application_layer.mnist_utils import *
except:
    from models import *
    from dataset_utils import *
    from mnist_utils import *

def get_model(model_str, dataset):
    """
    Returns a model of choice from the library, adapted also to the passed dataset
    :param model_str: the name of the required model
    :param dataset: the name of the dataset to be used
    :raises: ValueError
    :returns: Model object
    """
    num_class_dict = {'mnist':10, 'cifar10':10, 'cifar100':100, 'imagenet':1000}
    if dataset not in num_class_dict.keys():
        raise ValueError("Provided dataset ({}) is not known!".format(dataset))
    num_classes = num_class_dict[dataset]
    #Check if it is a native model or custom. The later should start with "my"
    if model_str.startswith('my'):
        #currently, we support: build_my_vgg, build_my_alexnet, build_my_inception, build_my_resnet, build_my_densenet
        model_str = model_str[2:]       #remove the "my" prefix
        if model_str.startswith('alex'):
            model = build_my_alexnet(num_classes)
        elif model_str.startswith('inception'):
            model = build_my_inception(num_classes)
        elif model_str.startswith('res'):
            model = build_my_resnet(model_str, num_classes)
        elif model_str.startswith('vgg'):
            model = build_my_vgg(model_str, num_classes)
        elif model_str.startswith('dense'):
            model = build_my_densenet(model_str, num_classes)
        else:
            ValueError("Provided model ({}) is not known!".format(model_str))
    else:
        models = {'convnet':Net,
		'cifarnet':Cifarnet,
		'cnn': CNNet,
                'alexnet': torchvision.models.alexnet,
		'resnet18':torchvision.models.resnet18,
                'resnet34':torchvision.models.resnet34,
                'resnet50':torchvision.models.resnet50,
                'resnet152':torchvision.models.resnet152,
		'inception':torchvision.models.inception_v3,
		'vgg16':torchvision.models.vgg16,
		'vgg19':torchvision.models.vgg19,
		'preactresnet18': PreActResNet18,
		'googlenet': GoogLeNet,
		'densenet121': torchvision.models.densenet121,
                'densenet201': torchvision.models.densenet201,
		'resnext29': ResNeXt29_2x64d,
		'mobilenet': MobileNet,
		'mobilenetv2': MobileNetV2,
		'dpn92': DPN92,
		'shufflenetg2': ShuffleNetG2,
		'senet18': SENet18,
		'efficientnetb0': EfficientNetB0,
		'regnetx200': RegNetX_200MF}
        if model_str not in models.keys():
            raise ValueError("Provided model ({}) is not known!".format(model_str))
        model = models[model_str](num_classes=num_classes)
    return model

def get_mem_usage():
    #returns a dict with memory usage values (in GBs)
    mem_dict = psutil.virtual_memory()
    all_mem = np.array([v for v in mem_dict])/(1024*1024*1024)	#convert to GB
    return {"available":all_mem[1],	"used":all_mem[3],
	"free":all_mem[4], "active":all_mem[5],
	"buffers":all_mem[7],"cached":all_mem[8]}

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def prepare_transforms(dataset_name):
  if dataset_name.startswith('cifar'):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_train, transform_test
  elif dataset_name == 'mnist':
    transform = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    return transform, transform
  elif dataset_name == 'imagenet':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])
    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])
    return transform_train, transform_test

def get_train_test_split(dataset_name, datadir, transform_train, transform_test):
  if dataset_name == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(
      root=datadir, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
      root=datadir, train=False, download=False, transform=transform_test)
  elif dataset_name == 'mnist':
    process_mnist(datadir)
    trainset = torchvision.datasets.MNIST(root=datadir, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.MNIST(root=datadir, train=False, download=False, transform=transform_test)
  elif dataset_name == 'imagenet':
    print("WARNING! Downloading the whole imagenet dataset is not recommended!")
    #First, we need to put images correctly in its folders
    os.system("cd {}; ./valprep.sh".format(os.path.join(datadir,'val')))
    #Then, we load the Imagenet dataset
    trainset = torchvision.datasets.ImageFolder(root=datadir, transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root=datadir, transform=transform_test)
  return trainset, testset

def stream_imagenet_batch(swift, datadir, parent_dir, labels, transform, batch_size, lstart, lend, model, mode='vanilla', split_idx=100, sequential=False):
  stream_time = time.time()
  if mode == 'split':
      post_step = 1000 		#if the batch is smaller, it will be handled on the server
      print("Start {}, end {}, post_step {}\r\n".format(lstart, lend, post_step))
      post_objects = []
      images = []
      post_time = time.time()
      for s in range(lstart, lend, post_step):
          opts = {"meta": {"Ml-Task:inference",
            "dataset:imagenet","model:{}".format(model),
            "Batch-Size:{}".format(int(post_step/10)),
            "start:{}".format(s),"end:{}".format(s+post_step),
            "Split-Idx:{}".format(split_idx)},
            "header": {"Parent-Dir:{}".format(parent_dir)}}
          obj_name = "{}/ILSVRC2012_val_000".format(parent_dir)+((5-len(str(s+1)))*"0")+str(s+1)+".JPEG"
          post_objects.append(SwiftPostObject(obj_name,opts))		#Create multiple posts
      post_time = time.time()
      read_bytes=0
      for post_res in swift.post(container='imagenet', objects=post_objects):
          if 'error' in post_res.keys():
              print("error: {}, traceback: {}".format(post_res['error'], post_res['traceback']))
          read_bytes += int(len(post_res['result']))
          print("Executing one post (whose result is of len {} bytes) took {} seconds".format(len(post_res['result']), time.time()-post_time))
          images.extend(pickle.loads(post_res['result']))			#images now should be a list of numpy arrays
      print("Read {} MBs for this batch".format(read_bytes/(1024*1024)))
      print("Executing all posts took {} seconds".format(time.time()-post_time))
      transform=None		#no transform required in this case
  else:		#mode=='vanilla'
    objects = []
    #prepare images that should be read
    for idx in range(lstart, lend):
      idstr = str(idx+1)
      obj_name = "{}/ILSVRC2012_val_000".format(parent_dir)+((5-len(idstr))*"0")+idstr+".JPEG"
      objects.append(obj_name)
    opts = {'out_directory':datadir}       #so that we can have it directly in memory
    #read all requested images
    images = []
    if sequential:
      queries = objects         #request them one by one then
    else:
      queries = swift.download(container='imagenet', objects=objects, options=opts)
    read_bytes=0
    for query in queries:
      if sequential:
        query = next(swift.download(container='imagenet', objects=[query], options=opts))
      read_bytes += int(query['read_length'])
      with open(os.path.join(datadir, query['object']), 'rb') as f:
        image_bytes = f.read()
      img = np.array(Image.open(BytesIO(image_bytes)).convert('RGB'))
      images.append(img)
    print("Read {} MBs for this batch".format(read_bytes/(1024*1024)))
  #use only labels corresponding to the required images
  labels = labels[lstart:lend]
  assert len(images) == len(labels)
  imgs = np.array(images)
  dataset = InMemoryDataset(imgs, labels=labels, transform=transform, mode=mode)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
  print("Streaming imagenet data took {} seconds".format(time.time()-stream_time))
  return dataloader

types = [torch.nn.modules.container.Sequential, _DenseLayer, _Transition]

globalFreezeIndex = 0
def freeze_sequential(network, all_layers):
    global globalFreezeIndex
    for layer in network.children():
        if type(layer) in types:
            freeze_sequential(layer, all_layers)
        else:
            all_layers.append(layer)
            if globalFreezeIndex >= len(all_layers):
                for param in layer.parameters():
                    param.requires_grad = False

def freeze_lower_layers(net, split_idx):
  #freeze the lower layers whose index < split_idx
  global globalFreezeIndex
  globalFreezeIndex = split_idx
  all_layers = []
  freeze_sequential(net, all_layers)
  #test if indeed frozen
#  total=0
#  frozen=0
#  for param in net.parameters():
#    if not param.requires_grad:
#      frozen+=1
#    total+=1
#  print("Total number of frozen layers: {} out of {} ".format(frozen,total))
