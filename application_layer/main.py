'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys
from utils import *
from mnist_utils import *
from time import time
from threading import Thread

from swiftclient.service import SwiftService, SwiftError
from swiftclient.exceptions import ClientException

def download_dataset(swift, dataset_name, datadir):
  """
  download dataset from swift db to local disk
	Parameters:
	swift                   Swift Service object to help download the dataset
        dataset_name (str)	name of the dataset to be downloaded (also the name of the container in swift db)
	datadir (str)		local directory on which the dataset should be stored
  """
  page_list = swift.list(container=dataset_name)
  opts = {'out_directory':datadir}
  for page in page_list:
    if page['success']:		#Otherwise, there is an error...skip
      objects = [obj['name'] for obj in page['listing']]
      for res in swift.download(container=dataset_name, objects=objects, options=opts):
        if not res['success']:
          print("ERROR!!", res)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--testonly', action='store_true', help='run inference only without training')
parser.add_argument('--downloadall', action='store_true',
                    help='download the whole dataset once in the beginning; recommended only with small datasets')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset to be used')
parser.add_argument('--model', default='convnet', type=str, help='model to be used')
parser.add_argument('--batch_size', default=100, type=int, help='batch size for dataloader')
parser.add_argument('--num_epochs', default=10, type=int, help='number of epochs for training')
parser.add_argument('--start', default=0, type=int, help='start index of data to be processed')
parser.add_argument('--end', default=10000, type=int, help='end index of data to be processed')
parser.add_argument('--split_idx', default=100, type=int, help='index at which computation is split between Swift and app. layer')
parser.add_argument('--freeze', action='store_true', help='freeze the lower layers of training model')
parser.add_argument('--sequential', action='store_true', help='execute in a single thread')
parser.add_argument('--cache', action='store_true', help='cache intermediate results into memory; useful in split mode')
args = parser.parse_args()

dataset_name = args.dataset
if not args.downloadall and (dataset_name == 'mnist' or dataset_name == 'cifar10'):
  print("WARNING: dataset {} is small enough! Will download it only once in the beginning!".format(dataset_name))
  args.downloadall = True

model = args.model
batch_size = args.batch_size
num_epochs = args.num_epochs
start = args.start
end = args.end
split_idx = args.split_idx
cache = args.cache
mode = 'split' if model.startswith("my") else 'vanilla'
print(args)

start_time = time()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')

homedir = os.environ['HOME']
datadir = os.path.join(homedir,"dataset",dataset_name)
#first fetch data....we assume here that data does not exist locally
swift = SwiftService()
if args.downloadall:
  start_download_t = time()
  try:
    download_dataset(swift, dataset_name, datadir)
  except ClientException as e:
    print("Got an exeption while downloading the dataset ", e)
  print('data downloaded...time elapsed: {}'.format(time()-start_download_t))

#prepare transformation
transform_train, transform_test = prepare_transforms(dataset_name)

if args.downloadall:
  trainset, testset = get_train_test_split(dataset_name, datadir, transform_train, transform_test)
  trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=batch_size, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(
      testset, batch_size=batch_size, shuffle=False, num_workers=2)

if not args.downloadall and dataset_name == 'imagenet':
  #We can download the labels file once and for all (it's small enough)
  opts = {'out_file':'-'}	#so that we can have it directly in memory
  query = swift.download(container=dataset_name, objects=['val/ILSVRC2012_validation_ground_truth.txt'], options=opts)
  reader = next(query)['contents']
  labelstr = b''.join(reader)
  labels = labelstr.decode("utf-8").split("\n")[:-1]		#remove extra '' at the end
  assert len(labels) == 150000		#remove this after making sure the code works
  labels = [int(l)-1 for l in labels]

# Model
print('==> Building model..')
net = get_model(model, dataset_name)
if mode == 'split' or args.freeze:
    print("Freezing the lower layers of the model ({}) till index {}".format(model, split_idx))
    freeze_lower_layers(net, split_idx)		#for transfer learning -- no need for backpropagation for upper layers (idx < split_idx)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)					#passing only parameters that require grad...the rest are frozen
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    global trainloader, net
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        if mode == 'split':		#This is transfer learning deceted!
            outputs = net(inputs, split_idx, 100)
        else:
            outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

def test(epoch):
    global testloader, net
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    res = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if mode == 'split':              #This is split inference deceted!
                outputs = net(inputs, split_idx, 100)
            else:
                outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            res.extend(predicted)
    return res

next_loader= None
def start_now(lstart, lend, transform):
  global next_dataloader
  next_dataloader = None
  next_dataloader = stream_imagenet_batch(swift, datadir, "val", labels, transform, batch_size, lstart, lend, model, mode, split_idx, args.sequential)

#step defines the number of images (or intermediate values) got from the server per iteration
#this value should be at least equal to the batch size
step = max(10000, batch_size)		#using a value less than 1000 is really waste of bandwidth (after some experimentation)
if cache and mode == 'vanilla':
  print("WARNING!! Cannot cache values in the vanila mode; will probably run out of memory\r\n Switching to non-cache mode")
  cache = False
if args.testonly:
  if not args.downloadall and dataset_name == 'imagenet':
    gstart, gend = start, end
    lstart, lend = gstart, gstart+step if gstart+step < gend else gend
    testloader = stream_imagenet_batch(swift, datadir, "val", labels, transform_test, batch_size, lstart, lend, model, mode, split_idx,args.sequential)
    res = []
    idx = 0
    for s in range(gstart+step, gend, step):
      lstart, lend = s,s+step if s+step < gend else gend
      myt = Thread(target=start_now, args=(lstart, lend,transform_test,))
      if not args.sequential:	#run this in parallel
        myt.start()
      lres = test(idx)
      res.extend(lres)
      idx+=1
      if args.sequential:
        myt.start()
      myt.join()
      testloader = next_dataloader
      dataloader = None
    res.extend(test(idx))
  else:
    res = test(0)
  print("Inference done for {} inputs".format(len(res)))
  print("The whole process took {} seconds".format(time()-start_time))
  exit(0)
else:
  if cache:
    dataloaders = []
  for epoch in range(num_epochs):
    cache_idx = 0
    if not args.downloadall and dataset_name == 'imagenet':
      lstart, lend = 0, step
      if cache and epoch != 0:		#This is the first epoch...I have to get data from Swift
        trainloader = dataloaders[cache_idx]		#first epoch, get the first batch
        cache_idx+=1
      else:
#        trainloader = stream_imagenet_batch(swift, datadir, "val", labels, transform_train, batch_size, lstart, lend, model, mode, split_idx,args.sequential)
        start_now(lstart, lend, transform_train)
        trainloader = next_dataloader
        if cache:
          dataloaders.append(trainloader)
      idx=0
      for s in range(step, 50000, step):
        localtime = time()
        lstart, lend = s, s+step
        if cache and epoch != 0:
          next_dataloader = dataloaders[cache_idx]
          cache_idx+=1
        else:
          myt = Thread(target=start_now, args=(lstart, lend,transform_train,))
          if not args.sequential:   #run this in parallel
            myt.start()
        train(epoch)
        print("One training iteration takes: {} seconds".format(time()-localtime))
        print("Index:",idx)
        idx+=1
        if args.sequential:
          if not cache or epoch == 0:
            myt.start()
        if not cache or epoch == 0:
          myt.join()
          if cache:
            dataloaders.append(next_dataloader)
        trainloader = next_dataloader
        dataloader = None
        print("Then, training+dataloading take {} seconds".format(time()-localtime))
      train(epoch)
    else:
      train(epoch)
    scheduler.step()
print("The whole process took {} seconds".format(time()-start_time))
sys.stdout.flush()