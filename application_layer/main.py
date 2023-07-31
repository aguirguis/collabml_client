'''Train CIFAR10 with PyTorch.'''
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
#parser.add_argument('--splitindex_to_freezeindex', action='store_true', help='If set, we use the freezing index as split point')
parser.add_argument('--split_choice', default='automatic', type=str, help='How to choose split_idx (manual, automatic, to_freeze, to_min, to_max)')


parser.add_argument('--cached', action='store_true', help='')
parser.add_argument('--transformed', action='store_true', help='')
parser.add_argument('--all_in_cos', action='store_true', help='')
parser.add_argument('--no_adaptation', action='store_true', help='')


start_time = time.time()


args = parser.parse_args()

dataset_name = args.dataset
if not args.downloadall and (dataset_name == 'mnist' or dataset_name == 'cifar10'):
  print("WARNING: dataset {} is small enough! Will download it only once in the beginning!".format(dataset_name))
  args.downloadall = True

stream_datasets = ['imagenet', 'plantleave', 'inaturalist']

stream_dataset_len = {'imagenet': 8000,#32000, #24320, #50000,
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



CACHED = args.cached
TRANSFORMED = args.transformed
ALL_IN_COS = args.all_in_cos
NO_ADAPT = args.no_adaptation

print("Cached ", CACHED)
print("Transformed ", TRANSFORMED)
print("All in COS ", ALL_IN_COS)
print("No adaptation ", NO_ADAPT)



if args.freeze and freeze_idx == 0:
  print("Freeze flag is set, but no freeze_idx was given! Will use the value of split_idx ({}) as a freeze_idx too!".format(split_idx))
  freeze_idx = split_idx
mode = 'split' if model.startswith("my") else 'vanilla'
print(args)

parent_dir = "compressed" # if mode == 'split' else "val"


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

print("Initialize time: {}".format(time.time()-start_time))

prepare_transforms_time = time.time()
#prepare transformation
transform_train, transform_test = prepare_transforms(dataset_name)
print("Time to prepare transforms: {}".format(time.time()-prepare_transforms_time))

if args.downloadall:
  trainset, testset = get_train_test_split(dataset_name, datadir, transform_train, transform_test)
  trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=batch_size, shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(
      testset, batch_size=batch_size, shuffle=False, num_workers=2)

download_labels_time = time.time()
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
print("Time to download labels: {}".format(time.time()-download_labels_time))

# Model
get_model_time = time.time()
print('==> Building model..')
net = get_model(model, dataset_name)
net.to(device)
print("Time for getting model: {}".format(time.time()-get_model_time))

mem_cons = [10,10]

model_prep_split = time.time()
predicted_sizes = []
times_for_prediction = {}
if mode == 'split':
    predicted_sizes, _, _, _ = get_intermediate_outputs_and_time(net, torch.rand((1, 3, 224, 224)).to(device))
    predicted_sizes = np.array(predicted_sizes)*1024.*batch_size # sizes is in Bytes (after *1024)
    times_for_prediction['inference_time'] = np.zeros(shape=(len(predicted_sizes)))
    split_choice_first = split_choice
    if split_choice == 'automatic':
        split_choice_first = 'to_freeze'
    split_idx, mem_cons = choose_split_idx(model, net, freeze_idx, batch_size, split_choice_first, split_idx, device, predicted_sizes)

print("Time for splitting algorithm: {}".format(time.time()-model_prep_split))

print(f"Using split index: {split_idx}")
freeze_model_time = time.time()
if mode == 'split' or args.freeze:
    if freeze_idx < split_idx and mode == 'split':
      print("WARNING! freeze_idx should be >= split_idx; setting freeze_idx to {}".format(split_idx))
      freeze_idx = split_idx
    print("Freezing the lower layers of the model ({}) till index {}".format(model, freeze_idx))
    freeze_lower_layers(net, freeze_idx)		#for transfer learning -- no need for backpropagation for upper layers (idx < split_idx)
    #freeze_lower_layers(list(net.children())[0], freeze_idx)		#for transfer learning -- no need for backpropagation for upper layers (idx < split_idx) -- if torch.nn-DataParallel(net) is done before

print("Time to freeze model: {}".format(time.time()-freeze_model_time))

prepare_model_time = time.time()
#if device == 'cuda':
##    torch.distributed.init_process_group(backend='nccl')
#    net = torch.nn.DataParallel(net)
#    cudnn.benchmark = True
#optimizer_time = time.time()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)					#passing only parameters that require grad...the rest are frozen
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#print("Time for criterion, optimizer, scheduler prep : {}".format(time.time()-optimizer_time))
print("Time for model prep: {}".format(time.time()-prepare_model_time))

print("Time for model and splitting algorithm: {}".format(time.time()-model_prep_split))


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
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_max_memory_allocated(i)
    train_times = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print("Time of next(dataloader) is: {}".format(time.time()-dataload_time))
        copy_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)
        print("Time for copying to cuda: {}".format(time.time()-copy_time))
        optimizer.zero_grad()
        forward_time = time.time()
        if mode == 'split':		#This is transfer learning deceted!
            outputs, _, i_train_times, _, _  = net(inputs, split_idx, 100, need_time=True)
            train_times.append(i_train_times)
        else:
            outputs = net(inputs)
        print("Time for forward pass: {}".format(time.time()-forward_time))
        back_time = time.time()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        backward_pass_time = time.time()-back_time
        print("Time for backpropagation: {}".format(backward_pass_time))
        if device == 'cuda':
            max_mem = 0.0
            for i in range(torch.cuda.device_count()):
                max_mem += torch.cuda.max_memory_allocated(i)
            print("GPU memory for training: {}         \
                 \r\n".format((max_mem)/(1024*1024*1024)))
        dataload_time = time.time()
    return np.mean(train_times, axis=0), backward_pass_time

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
            if mode == 'split':              #This is split inference detected!
                outputs,_,_,_,_ = net(inputs, split_idx, 100, need_time=True)
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
def start_now(lstart, lend, transform, result=[], do_split=False):
  global next_dataloader
  global times_for_prediction
  next_dataloader = None
  

  split_idx_ = split_idx
  mem_cons_ = mem_cons
  if do_split:
      split_idx_, mem_cons_ = choose_split_idx(model, net, freeze_idx, batch_size, split_choice, split_idx, device, predicted_sizes, times_for_prediction)

  next_dataloader, server_times = stream_batch(dataset_name, stream_dataset_len, swift, datadir, parent_dir, labels, transform, batch_size, lstart, lend, model, mode, split_idx_, mem_cons_,  args.sequential, CACHED, TRANSFORMED, ALL_IN_COS, NO_ADAPT)

  #print('Server times :', server_times)


  times_for_prediction['s_read_data_shm'] = server_times['s_read_data_shm']
  times_for_prediction['s_read_model_shm_to_gpu'] = server_times['s_read_model_shm_to_gpu']
  
  nb_inferences = batch_size/SERVER_BATCH
  times_for_prediction['s_inf_to_pytorch'] = server_times['s_inf_to_pytorch']*nb_inferences
  times_for_prediction['s_inf_copy_to_gpu'] = server_times['s_inf_copy_to_gpu']*nb_inferences
  times_for_prediction['s_inf_to_numpy'] = server_times['s_inf_to_numpy']*nb_inferences

  for idx in range(len(server_times['s_inf_forward_pass'])):
      times_for_prediction['inference_time'][idx] = server_times['s_inf_forward_pass'][idx]*nb_inferences

  #print("Times for prediction : ", times_for_prediction)

  result.append((split_idx_, mem_cons_))

#step defines the number of images (or intermediate values) got from the server per iteration
#this value should be at least equal to the batch size
step = batch_size #max(2000, batch_size)		#using a value less than 1000 is really waste of bandwidth (after some experimentation)
training_streaming_start_time = time.time()
try:
  if args.testonly:
    if not args.downloadall and dataset_name in stream_datasets:
      gstart, gend = start, end
      lstart, lend = gstart, gstart+step if gstart+step < gend else gend
      testloader, _ = stream_batch(dataset_name, stream_dataset_len, swift, datadir, parent_dir, labels, transform_test, batch_size, lstart, lend, model, mode, split_idx, mem_cons, args.sequential, CACHED, TRANSFORMED, ALL_IN_COS, NO_ADAPT)
      res = []
      idx_iter = 0
      for s in range(gstart+step, gend, step):
        results_split_idx_mem_cons = []
        lstart, lend = s,s+step if s+step < gend else gend
        myt = Thread(target=start_now, args=(lstart, lend,transform_test,results_split_idx_mem_cons))
        if not args.sequential:	#run this in parallel
          myt.start()
        lres = test(idx_iter)
        res.extend(lres)
        idx_iter+=1
        if args.sequential:
          myt.start()
        myt.join()
        testloader = next_dataloader
        dataloader = None
      res.extend(test(idx_iter))
    else:
      res = test(0)
    print("Inference done for {} inputs".format(len(res)))
    print("The whole process took {} seconds".format(time.time()-start_time))
  else:
    for epoch in range(num_epochs):
      if not args.downloadall and dataset_name in stream_datasets:
        lstart, lend = 0, step
        first_stream = time.time()
        trainloader, _ = stream_batch(dataset_name, stream_dataset_len, swift, datadir, parent_dir, labels, transform_train, batch_size, lstart, lend, model, mode, split_idx,mem_cons, args.sequential, CACHED, TRANSFORMED, ALL_IN_COS, NO_ADAPT)
        print("First stream takes: {} seconds".format(time.time()-first_stream))
        idx_iter=0
        for s in range(step, stream_dataset_len[dataset_name], step):
          print("Index: ", idx_iter)
          print("Split idx: ", split_idx)
          results_split_idx_mem_cons = []
          localtime = time.time()
          lstart, lend = s, s+step
          if idx_iter == 2:
            myt = Thread(target=start_now, args=(lstart, lend,transform_train,results_split_idx_mem_cons, True))
          else:
            myt = Thread(target=start_now, args=(lstart, lend,transform_train,results_split_idx_mem_cons))
          if not args.sequential:   #run this in parallel
            myt.start()
          train_times, backward_pass_time = train(epoch)
          
          for idx in range(split_idx, split_idx+len(train_times)):
            times_for_prediction['inference_time'][idx] = train_times[(idx-split_idx)]
          print(times_for_prediction['inference_time'])
          times_for_prediction['c_train_backward_pass'] = backward_pass_time

          print("One training iteration takes: {} seconds".format(time.time()-localtime))
          idx_iter+=1
          if args.sequential:
            myt.start()
          myt.join()
          
          split_idx, mem_cons = results_split_idx_mem_cons[0]

          trainloader = next_dataloader
          dataloader = None
          print("Then, training+dataloading take {} seconds".format(time.time()-localtime))
          
        last_train = time.time()
        train(epoch)
        print("Last train takes: {} seconds".format(time.time()-last_train))
      else:
        train(epoch)
      scheduler.step()
except Exception as e:
  import traceback
  traceback.print_exc()
  print(f"Exception: {e}")
  #for c in psutil.Process(os.getpid()).children(recursive=True):
  #    c.kill()


print("Total time for streaming and training: {}".format(time.time()-training_streaming_start_time))

gpu_periodic_stat_finish_time = time.time()
#Stop GPU thread
stop_thread = True
gpu_th.join()

print("Finish gpu thread time: {}".format(time.time()-gpu_periodic_stat_finish_time))

print("The whole process took {} seconds".format(time.time()-start_time))
sys.stdout.flush()
