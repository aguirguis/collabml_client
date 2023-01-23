'''Train CIFAR10 with PyTorch.'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse
import sys
from utils import *
from mnist_utils import *
import time
from threading import Thread

from swiftclient.service import SwiftService, SwiftError
from swiftclient.exceptions import ClientException

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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
parser.add_argument('--freeze_idx', default=0, type=int, help='index at which network is frozen (for transfer learning)')
parser.add_argument('--freeze', action='store_true', help='freeze the lower layers of training model')
parser.add_argument('--sequential', action='store_true', help='execute in a single thread')
parser.add_argument('--cpuonly', action='store_true', help='do not use GPUs')
parser.add_argument('--use_intermediate', action='store_true', help='If set, we use intermediate compute server between Swift server and client. Otherwise, ML computation (i.e., feature extraction) will happen inside Swift')
#parser.add_argument('--splitindex_to_freezeindex', action='store_true', help='If set, we use the freezing index as split point')
parser.add_argument('--split_choice', default='automatic', type=str, help='How to choose split_idx (manual, automatic, to_freeze, to_min, to_max)')
args = parser.parse_args()

dataset_name = args.dataset
if not args.downloadall and (dataset_name == 'mnist' or dataset_name == 'cifar10'):
  print("WARNING: dataset {} is small enough! Will download it only once in the beginning!".format(dataset_name))
  args.downloadall = True

stream_datasets = ['imagenet', 'plantleave', 'inaturalist']

stream_dataset_len = {'imagenet': 32000, #24320, #50000,
            'plantleave': 4502,
            'inaturalist': 24426
        }

model = args.model
batch_size = args.batch_size
num_epochs = args.num_epochs
start = args.start
end = args.end
split_idx = args.split_idx
freeze_idx = args.freeze_idx
split_choice = args.split_choice
if args.freeze and freeze_idx == 0:
  print("Freeze flag is set, but no freeze_idx was given! Will use the value of split_idx ({}) as a freeze_idx too!".format(split_idx))
  freeze_idx = split_idx
mode = 'split' if model.startswith("my") else 'vanilla'
print(args)

parent_dir = "compressed" # if mode == 'split' else "val"

start_time = time.time()

print("Nb GPUS ", torch.cuda.device_count())

device = 'cuda' if torch.cuda.is_available() and not args.cpuonly else 'cpu'


# Data
print('==> Preparing data..')

homedir = os.environ['HOME']
datadir = os.path.join(homedir,"dataset",dataset_name)
#first fetch data....we assume here that data does not exist locally
swift = SwiftService()
if args.downloadall:
  start_download_t = time.time()
  try:
    download_dataset(swift, dataset_name, datadir)
  except ClientException as e:
    print("Got an exeption while downloading the dataset ", e)
  print('data downloaded...time elapsed: {}'.format(time.time()-start_download_t))

#prepare transformation
transform_train, transform_test = prepare_transforms(dataset_name)

if args.downloadall:
  trainset, testset = get_train_test_split(dataset_name, datadir, transform_train, transform_test)
  trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=batch_size, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(
      testset, batch_size=batch_size, shuffle=False, num_workers=2)

if not args.downloadall and dataset_name in stream_datasets:
  #We can download the labels file once and for all (it's small enough)
  opts = {'out_file':'-'}	#so that we can have it directly in memory
  query = swift.download(container=dataset_name, objects=['compressed/ILSVRC2012_validation_ground_truth.txt'], options=opts)
  #print(query)
  reader = next(query)['contents']
  labelstr = b''.join(reader)
  labels = labelstr.decode("utf-8").split("\n")[:-1]		#remove extra '' at the end
  if dataset_name == 'imagenet':
    assert len(labels) == 150000		#remove this after making sure the code works
    labels = [int(l)-1 for l in labels]

# Model
print('==> Building model..')
net = get_model(model, dataset_name)
mem_cons = [10,10]
if mode == 'split':
    split_idx, mem_cons = choose_split_idx(model, net, freeze_idx, batch_size, split_choice, split_idx, device)

print(f"Using split index: {split_idx}")
if mode == 'split' or args.freeze:
    if freeze_idx < split_idx and mode == 'split':
      print("WARNING! freeze_idx should be >= split_idx; setting freeze_idx to {}".format(split_idx))
      freeze_idx = split_idx
    print("Freezing the lower layers of the model ({}) till index {}".format(model, freeze_idx))
    freeze_lower_layers(net, freeze_idx)		#for transfer learning -- no need for backpropagation for upper layers (idx < split_idx)

net = net.to(device)
if device == 'cuda':
#    torch.distributed.init_process_group(backend='nccl')
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
    dataload_time = time.time()
    if device == 'cuda':
        torch.cuda.reset_max_memory_allocated(0)
        torch.cuda.reset_max_memory_allocated(1)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print("Time of next(dataloader) is: {}".format(time.time()-dataload_time))
        copy_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        print("Time for copying to cuda: {}".format(time.time()-copy_time))
        optimizer.zero_grad()
        forward_time = time.time()
        if mode == 'split':		#This is transfer learning deceted!
            outputs,_ = net(inputs, split_idx, 100)
        else:
            outputs = net(inputs)
        print("Time for forward pass: {}".format(time.time()-forward_time))
        back_time = time.time()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print("Time for backpropagation: {}".format(time.time()-back_time))
        if device == 'cuda':
            print("GPU memory for training: {}         \
                 \r\n".format((torch.cuda.max_memory_allocated(0)+torch.cuda.max_memory_allocated(1))/(1024*1024*1024)))
        dataload_time = time.time()

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
                outputs,_,_ = net(inputs, split_idx, 100)
            else:
                outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            res.extend(predicted)
    return res

#Create and start the thread of monitoring GPU memory consumption
stop_thread = False
gpu_th = Thread(target=get_periodic_stat, args=(lambda: stop_thread, ))
gpu_th.start()

next_loader= None
def start_now(lstart, lend, transform):
  global next_dataloader
  next_dataloader = None
  # TODO run splitting algo at each iteration
  #split_idx, mem_cons = choose_split_idx(net, freeze_idx, batch_size, split_choice, split_idx)

  next_dataloader = stream_batch(dataset_name, stream_dataset_len, swift, datadir, parent_dir, labels, transform, batch_size, lstart, lend, model, mode, split_idx, mem_cons,  args.sequential, args.use_intermediate)

#step defines the number of images (or intermediate values) got from the server per iteration
#this value should be at least equal to the batch size
step = batch_size #max(2000, batch_size)		#using a value less than 1000 is really waste of bandwidth (after some experimentation)
try:
  if args.testonly:
    if not args.downloadall and dataset_name in stream_datasets:
      gstart, gend = start, end
      lstart, lend = gstart, gstart+step if gstart+step < gend else gend
      testloader = stream_batch(dataset_name, stream_dataset_len, swift, datadir, parent_dir, labels, transform_test, batch_size, lstart, lend, model, mode, split_idx, mem_cons, args.sequential,args.use_intermediate)
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
    print("The whole process took {} seconds".format(time.time()-start_time))
  else:
    for epoch in range(num_epochs):
      if not args.downloadall and dataset_name in stream_datasets:
        lstart, lend = 0, step
        trainloader = stream_batch(dataset_name, stream_dataset_len, swift, datadir, parent_dir, labels, transform_train, batch_size, lstart, lend, model, mode, split_idx,mem_cons, args.sequential, args.use_intermediate)
        idx=0
        for s in range(step, stream_dataset_len[dataset_name], step):
        #for s in range(step, 24320, step):			#TODO: Here, replace 50000 with step if you want to run 1 iteration only
        #for s in range(step, 24000, step):			#TODO: Here, replace 50000 with step if you want to run 1 iteration only
          localtime = time.time()
          lstart, lend = s, s+step
          myt = Thread(target=start_now, args=(lstart, lend,transform_train,))
          if not args.sequential:   #run this in parallel
            myt.start()
          train(epoch)
          print("One training iteration takes: {} seconds".format(time.time()-localtime))
          print("Index:",idx)
          idx+=1
          if args.sequential:
            myt.start()
          myt.join()
          trainloader = next_dataloader
          dataloader = None
          print("Then, training+dataloading take {} seconds".format(time.time()-localtime))
        train(epoch)
      else:
        train(epoch)
      scheduler.step()
except Exception as e:
  import traceback
  traceback.print_exc()
  print(f"Exception: {e}")
#Stop GPU thread
stop_thread = True
gpu_th.join()
print("The whole process took {} seconds".format(time.time()-start_time))
sys.stdout.flush()
