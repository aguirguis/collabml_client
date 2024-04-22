'''Train CIFAR10 with PyTorch.'''
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
import signal
import math
import numpy as np

from multiprocessing import Queue, Process, shared_memory

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import subprocess

np.set_printoptions(linewidth=np.inf)

def signal_handler(sig, frame):
    print ("Signal handler called at", time.time(), flush=True)
    time.sleep(1)
    os._exit(1) 


def train(epoch):
    global trainloader, net
    #print('\nEpoch: %d' % epoch)
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
        if mode == 'split':   #This is transfer learning deceted!
            outputs, _ , i_train_times, _, _  = net(inputs, split_idx, 100, need_time=True)
            train_times.append(i_train_times)
        else:
            outputs, _ , i_train_times,_,_ = net(inputs, need_time=True)
            train_times.append(i_train_times)
        print("Time for forward pass: {}".format(time.time() - forward_time))
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
                 ".format((max_mem)/(1024*1024*1024)))
        dataload_time = time.time()
    if len(train_times) > 0:
        return np.mean(train_times, axis=0), backward_pass_time
    else:
        return 0,0


def update_times_for_prediction(server_times): 
  global times_for_prediction
  #nb_inferences = batch_size/SERVER_BATCH
  #nb_inferences = 8 * int(math.ceil(batch_size/512))
  nb_inferences = 8 #* min(4, batch_size/128)
  #print("Calculating nb_inferences as: ",nb_inferences, batch_size, SERVER_BATCH)

  times_for_prediction['s_read_data_shm'] = server_times['s_read_data_shm']
  times_for_prediction['s_read_model_shm_to_gpu'] = server_times['s_read_model_shm_to_gpu']

  times_for_prediction['s_inf_to_pytorch'] = server_times['s_inf_to_pytorch']*nb_inferences
  times_for_prediction['s_inf_copy_to_gpu'] = server_times['s_inf_copy_to_gpu']*nb_inferences
  times_for_prediction['s_inf_to_numpy'] = server_times['s_inf_to_numpy']*nb_inferences
  #times_for_prediction['futures'] = server_times['futures']
  #times_for_prediction['total_before_send'] = server_times['total_before_send']
  #times_for_prediction['transferred'] = server_times['transferred']
  times_for_prediction['server_bw'] = times_for_prediction['server_bw'] + server_times['server_bw']
  times_for_prediction['server_bw_count'] = times_for_prediction['server_bw_count'] + 1
  times_for_prediction['server_fwd_pass_count'] = times_for_prediction['server_fwd_pass_count'] + 1
  print("Server sent data with bandwidth: ", server_times['server_bw'])


  print("len(server_times['s_inf_forward_pass']: ",len(server_times['s_inf_forward_pass']))
  for idx in range(len(server_times['s_inf_forward_pass'])):
      times_for_prediction['inference_time'][idx] = (times_for_prediction['inference_time'][idx] * ( times_for_prediction['server_fwd_pass_count'] - 1 ) + server_times['s_inf_forward_pass'][idx] * nb_inferences) * 1.0 / times_for_prediction['server_fwd_pass_count']
      times_for_prediction['inference_time2'][idx] = (times_for_prediction['inference_time2'][idx] * ( times_for_prediction['server_fwd_pass_count'] - 1 ) + server_times['avg_srv_times'][idx] * nb_inferences) * 1.0 / times_for_prediction['server_fwd_pass_count']

  print("Times for prediction : ", times_for_prediction)
  print("Len times for prediction[inference_time] : ", len(times_for_prediction['inference_time']))



def start_now_subprocess3(in_queue, out_queue, transform, shm):

  while True:
    print ("Subprocess waiting to start read at: ", time.time(),flush=True)
    lstart = in_queue.get()
    lend   = in_queue.get()
    split_ = in_queue.get()
    mode = in_queue.get()
    print ("Subprocess got parameters to start read at: ", time.time(),flush=True)

    if (lstart == -1):
        print ("Subprocess finishes at: ",time.time())
        break
    
    start_now_subprocess2(out_queue, lstart, lend, transform, shm, split_, mode)


  

def start_now_subprocess2(queue, lstart, lend, transform, shm, split_idx, mode):

  print ("Subprocess run for range ",lstart," ",lend," split ",split_idx," at: ", time.time(), flush=True)

  _, server_times, len_read, indexes = stream_batch(dataset_name, stream_dataset_len, swift, datadir, parent_dir, labels, transform, batch_size, lstart, lend, model, SERVER_IP, shm, mode, split_idx, server_batch_size, None , args.sequential, CACHED, TRANSFORMED, ALL_IN_COS, NO_ADAPT)

  print("Subprocess adds to mp.queue at: ", time.time(), flush=True)
  queue.put(server_times)
  queue.put(len_read)
  queue.put(indexes)
  print("Subprocess done adding to mp.queue at: ", time.time(), flush=True)



def sync_with_ioprocess_and_generate_dl(q1, lstart, lend, mode):
  if (mode == 'split'):
    return sync_with_ioprocess_and_generate_dl_hapi(q1, lstart, lend)   
  return sync_with_ioprocess_and_generate_dl_baseline(q1, lstart, lend)  



def sync_with_ioprocess_and_generate_dl_baseline(q1, lstart, lend):
  print("Before reading from mp.queue at: ", time.time(),flush=True)
  server_times=q1.get()
  len_read = q1.get()
  indexes = q1.get()
  print(indexes)
  print("After reading from mp.queue at: ", time.time(), flush=True)
                      
  len_read_mb = int(len_read/1024/1024)

  images = []

  pickle_start_time = time.time()
  lend = stream_dataset_len[dataset_name] if lend > stream_dataset_len[dataset_name] else lend
  nr_req = int((lend-lstart)/128)
  #len_read_per_req = int(len_read/nr_req)
  start_read = 0

  time_decompress_start=time.time()
  for i in range(nr_req):
    #shm_trunc = shm.buf[len_read_per_req * i : len_read_per_req * ]
    shm_trunc=shm.buf[start_read: start_read + indexes[i]]
    f2 = BytesIO(shm_trunc)
    zipff = zipfile.ZipFile(f2, 'r')
    #print ("XXXXX")
    #print (zipff.infolist())
    #print (zipff)
  #print("Decompress took {} sec for {} MB thr {} at: {}".format(decompress_duration, len_read_mb, int(len_read_mb/decompress_duration), time.time()), flush=True)

  #time_other_start=time.time()
    if not TRANSFORMED:
      images.extend(np.array(Image.open(io.BytesIO(zipff.open(f3).read())).convert('RGB')) for f3 in zipff.infolist())
      transform=transform_train
    else:
      images.extend(np.array(torch.load(io.BytesIO(zipff.open(f3).read()))) for f3 in zipff.infolist())
      transform=None
    start_read = start_read + indexes[i]


  #print("Other took {} sec for {} MB thr {} at: {}".format(decompress_duration, len_read_mb, int(len_read_mb/decompress_duration), time.time()), flush=True)
  decompress_duration = time.time() - time_decompress_start
  print("Decompress took {} sec for {} MB thr {} at: {}".format(decompress_duration, len_read_mb, int(len_read_mb/decompress_duration), time.time()), flush=True)

  labels_trunc = labels[lstart:lend]
  #print("len_images: ", len(images), " len_labels: ", len(labels_trunc))
  assert len(images) == len(labels_trunc)
    
  dataloader_start_time = time.time()
  #imgs = np.array(images)
  imgs=images
  #transform = None
  dataset = InMemoryDataset(imgs, labels=labels, transform=transform, mode=mode, transformed=TRANSFORMED)
  next_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
  dl_duration = time.time() - dataloader_start_time

  print("Dataloader took {} sec for {} MB thr {} at: {}".format(dl_duration, len_read_mb, int(len_read_mb/dl_duration),time.time()), flush=True)
  return next_dataloader




def sync_with_ioprocess_and_generate_dl_hapi(q1, lstart, lend):
  print("Before reading from mp.queue at: ", time.time(),flush=True)
  server_times=q1.get()
  len_read = q1.get()
  indexes = q1.get()
  print("After reading from mp.queue at: ", time.time(), flush=True)
  update_times_for_prediction(server_times)
                      
  len_read_mb = int(len_read/1024/1024)

  images = []

  pickle_start_time = time.time()
  lend = stream_dataset_len[dataset_name] if lend > stream_dataset_len[dataset_name] else lend
  nr_req = int((lend-lstart)/128)
  len_read_per_req = int(len_read/nr_req)
  for i in range(nr_req):
    shm_trunc = shm.buf[len_read_per_req * i : len_read_per_req * (i + 1)]
    images.extend(pickle.loads(shm_trunc))
    #print("After unpickling round ", i + 1, " size of images is: ", len(images))
  pickle_duration = time.time() - pickle_start_time
  print("Pickle took {} sec for {} MB thr {} at: {}".format(pickle_duration, len_read_mb, int(len_read_mb/pickle_duration), time.time()), flush=True)

  labels_trunc = labels[lstart:lend]
  #print("len_images: ", len(images), " len_labels: ", len(labels_trunc))
  assert len(images) == len(labels_trunc)
    
  dataloader_start_time = time.time()
  imgs = np.array(images)
  transform = None
  dataset = InMemoryDataset(imgs, labels=labels, transform=transform, mode=mode, transformed=TRANSFORMED)
  next_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
  dl_duration = time.time() - dataloader_start_time

  print("Dataloader took {} sec for {} MB thr {} at: {}".format(dl_duration, len_read_mb, int(len_read_mb/dl_duration),time.time()), flush=True)
  return next_dataloader





#####################################################################################################################


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
parser.add_argument('--end0', default=10000, type=int, help='end index of data to be processed')
parser.add_argument('--split_idx', default=100, type=int, help='index at which computation is split between Swift and app. layer')
parser.add_argument('--freeze_idx', default=0, type=int, help='index at which network is frozen (for transfer learning)')
parser.add_argument('--freeze', action='store_true', help='freeze the lower layers of training model')
parser.add_argument('--sequential', action='store_true', help='execute in a single thread')
parser.add_argument('--cpuonly', action='store_true', help='do not use GPUs')
#parser.add_argument('--splitindex_to_freezeindex', action='store_true', help='If set, we use the freezing index as split point')
parser.add_argument('--split_choice', default='automatic', type=str, help='How to choose split_idx (manual, automatic, to_freeze, to_min, to_max)')
parser.add_argument('--gpus', default=1, type=int, help='nr gpus')
parser.add_argument('--sip', default="SERVER_IP", type=str, help='')
parser.add_argument('--srvbs', default=16, type=int, help='server batch size')
parser.add_argument('--nwbw', default=16384, type=int, help='kbps network')

parser.add_argument('--cached', action='store_true', help='')
parser.add_argument('--transformed', action='store_true', help='')
parser.add_argument('--all_in_cos', action='store_true', help='')
parser.add_argument('--no_adaptation', action='store_true', help='')


start_time = time.time()
last_epoch_start = time.time()

args = parser.parse_args()
SERVER_IP = args.sip

dataset_name = args.dataset

if args.gpus == 1:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
elif args.gpus == 2:
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
else:
    print ("Not a valid number of GPUS");


stream_datasets = ['imagenet', 'plantleave', 'inaturalist']

stream_dataset_len = {'imagenet': args.end,#32000, #24320, #50000,
            'plantleave': 4502,
            'inaturalist': 24426
        }

model = args.model
batch_size = args.batch_size
num_epochs = args.num_epochs
start = args.start
end = args.end
end0 = args.end0
split_idx = args.split_idx
freeze_idx = args.freeze_idx
split_choice = args.split_choice
server_batch_size=args.srvbs
nwbw=args.nwbw

signal.signal(signal.SIGINT, signal_handler)


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

print("Initialize time: {}".format(time.time()-start_time))

prepare_transforms_time = time.time()
#prepare transformation
transform_train, transform_test = prepare_transforms(dataset_name)
print("Time to prepare transforms: {}".format(time.time()-prepare_transforms_time))


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

GPU_in = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
net.to(device)
model_size = torch.cuda.max_memory_allocated(device) / (1024 ** 2) - GPU_in 

print("Time for getting model: {}".format(time.time()-get_model_time))

mem_cons = [10,10]

model_prep_split = time.time()
predicted_sizes = []
times_for_prediction1 = {}
times_for_prediction2 = {}
times_for_prediction = times_for_prediction1
times_for_prediction1['server_bw_count']=0
times_for_prediction1['server_bw']=0
times_for_prediction1['server_fwd_pass_count']=0
times_for_prediction2['server_bw_count']=0
times_for_prediction2['server_bw']=0
times_for_prediction2['server_fwd_pass_count']=0

if mode == 'split':
    predicted_sizes, _, _, _ = get_intermediate_outputs_and_time(net, torch.rand((1, 3, 224, 224)).to(device))
    predicted_sizes = np.array(predicted_sizes)*1024.*batch_size # sizes is in Bytes (after *1024)
    times_for_prediction1['inference_time'] = np.zeros(shape=(len(predicted_sizes)))
    times_for_prediction1['inference_time2'] = np.zeros(shape=(len(predicted_sizes)))
    times_for_prediction2['inference_time'] = np.zeros(shape=(len(predicted_sizes)))
    times_for_prediction2['inference_time2'] = np.zeros(shape=(len(predicted_sizes)))
    split_choice_first = split_choice
    if split_choice == 'automatic':
        split_choice_first = 'to_freeze'
    split_idx, mem_cons, _ = choose_split_idx(model, net, freeze_idx, batch_size, split_choice_first, split_idx, nwbw, device, predicted_sizes, -1, {}, {})

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

split_old = split_idx
split_new = split_idx

print("Time to freeze model: {}".format(time.time()-freeze_model_time))

prepare_model_time = time.time()
if device == 'cuda': 
    #torch.distributed.init_process_group(backend='nccl')
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
#optimizer_time = time.time()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)					#passing only parameters that require grad...the rest are frozen
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
#print("Time for criterion, optimizer, scheduler prep : {}".format(time.time()-optimizer_time))
print("Time for model prep: {}".format(time.time()-prepare_model_time))

print("Time for model and splitting algorithm: {}".format(time.time()-model_prep_split))


#for bs in [128,256,512,1024,1536,2048,3072]:
    #for sidx in range(1,freeze_idx):

#early_profiling_split is either 0 (baseline) or > 0 (split)
oom_splits, early_profiling_split = estimate_memory(model, net, freeze_idx, batch_size, 999, device, model_size)

print("torch.cuda.max_memory_allocated - aft estimations",torch.cuda.max_memory_allocated(device) / (1024 ** 2))
print("Splits that we estimate will OOM :",oom_splits)


#Create and start the thread of monitoring GPU memory consumption
stop_thread = False
gpu_th = Thread(target=get_periodic_stat, args=(lambda: stop_thread, ))
gpu_th.start()

next_loader= None


SHM_SIZE=10 * 1024 * 1024 * 1024
SHM_NAME="shm"  
ts_create_shm = time.time()
try:
  shm = shared_memory.SharedMemory(create=True, size=SHM_SIZE, name=SHM_NAME)
except FileExistsError:
  shm = shared_memory.SharedMemory(create=False, size=SHM_SIZE, name=SHM_NAME)
print("Creating shm took: ", time.time() - ts_create_shm)

q = Queue()
q1 = Queue()
p = Process(target=start_now_subprocess3, args=(q,q1,transform_train, SHM_NAME))
p.start()


#labels=None
#step defines the number of images (or intermediate values) got from the server per iteration
#this value should be at least equal to the batch size
step = batch_size #max(2000, batch_size)		#using a value less than 1000 is really waste of bandwidth (after some experimentation)
training_streaming_start_time = time.time()
last_it_time = time.time()
print("torch.cuda.max_memory_allocated - bef loop",torch.cuda.max_memory_allocated(device) / (1024 ** 2))
did_switch=0
vanilla_it=0
try:
  if args.testonly:
    print ("Path not taken")  
  else:
    temporary_switch=False
    for epoch in range(num_epochs):
      print ("Starting epoch ",epoch, " at time: ", time.time())
      
      last_epoch_start=time.time()

      if not args.downloadall and dataset_name in stream_datasets:
        lstart, lend = 0, step
   
        q.put(lstart)
        q.put(lend)
        q.put(split_idx)
        q.put(mode)
        trainloader = sync_with_ioprocess_and_generate_dl(q1, lstart, lend, mode)

        idx_iter=0
        ending = stream_dataset_len[dataset_name]
        if (epoch == 0):
            ending = args.end0


        est_baseline_count=0
        est_baseline_total=0
        for s in range(step, ending, step):
          print("\nEpoch: ", epoch, "iteration: ", idx_iter, "split_idx: ", split_idx , "at: ", time.time(), flush=True)

          localtime = time.time()
          lstart, lend = s, s+step

          switched2 = 'none'
          if (mode == 'split' or temporary_switch == True) and lend >= ending and epoch == 0:
            print("Before killing process:", time.time(), flush=True)
            subprocess.run(["pkill","--signal","SIGTERM","-f","tcpdump"])
            print("After killing process:", time.time(), flush=True)

            if est_baseline_count != 0:   
              est_baseline_rt = (est_baseline_total/est_baseline_count) * (30720/batch_size)   
            else:
              est_baseline_rt=100000   
            print("QQQ baseline runtime: ", est_baseline_rt)
            print (times_for_prediction1)
            print (times_for_prediction2)
            split_idx, mem_cons, split_idx_rt = choose_split_idx(model, net, freeze_idx, batch_size, split_choice, split_idx, nwbw, device, predicted_sizes, oom_splits, early_profiling_split, times_for_prediction1, times_for_prediction2)
            if (est_baseline_rt < split_idx_rt and split_choice == 'automatic'):
              split_idx = early_profiling_split   #you are lucky here that the last part of mixed profiling was vanilla as well
              split_new = split_idx
              temporary_switch=True
              switched2 = 'vanilla'
            else:
              split_new = split_idx
              temporary_switch = False
              switched2 = 'split'
            print("QQMP Final mix-profiling switching at iteration: ", idx_iter, " splits: ", split_old, " -> ", split_idx)
              

          switched='none'
          if mode == 'split' and lend >= ending/2 and epoch == 0 and did_switch == 0 and (split_choice == 'automatic' or split_choice == 'nsg'):          
            print("QQMP Mid-point mix-profiling switching at iteration: ", idx_iter, " splits: ", split_idx, " -> ", early_profiling_split)
            split_idx = early_profiling_split
            split_new = split_idx
            switched='split'
            did_switch=1
            if(early_profiling_split == 0): 
              temporary_switch=True
              switched = 'vanilla'

          q.put(lstart)
          q.put(lend)
          q.put(split_new)  
          if 'none' not in switched2:
            q.put(switched2)
          elif  'none' not in switched:
            q.put(switched)
          else:
            q.put(mode)


          split_idx = split_old
          print("Training on client started at: ",time.time(), flush=True)
          train_times, backward_pass_time = train(epoch)
          train_finished_time = time.time()
          print("Training on client finished at: ",time.time(), flush=True)
          print("torch.cuda.max_memory_allocated - aft train",torch.cuda.max_memory_allocated(device) / (1024 ** 2))
          print("One training iteration takes: {} seconds at {} total {}".format(time.time()-localtime, time.time(),time.time()-last_it_time))
          
          est_baseline_rt = time.time() - last_it_time

          last_it_time = time.time()
          idx_iter+=1
          if temporary_switch == True:
            if (vanilla_it > 2):
              est_baseline_total += est_baseline_rt
              print("Adding to estimate: ", est_baseline_rt)
              est_baseline_count += 1
            vanilla_it += 1

          #so this is not an average?  
          if mode == 'split' or temporary_switch == True: 
              for idx in range(split_idx, split_idx + len(train_times)):
                times_for_prediction['inference_time'][idx] = train_times[(idx-split_idx)]
                times_for_prediction['inference_time2'][idx] = train_times[(idx-split_idx)]
              times_for_prediction['c_train_backward_pass'] = backward_pass_time

          split_idx = split_new
          split_old = split_new

          if 'none' not in switched2:
            mode = switched2
          if 'none' not in switched:
            mode = switched
            times_for_prediction = times_for_prediction2

         
          next_dataloader = sync_with_ioprocess_and_generate_dl(q1, lstart, lend, mode)
    

          trainloader = next_dataloader
          dataloader = None
          
          print("Then, training+dataloading take {} seconds at: {}".format(time.time()-localtime, time.time()))



        print("\nEpoch: ", epoch, "iteration: ", idx_iter, "split_idx: ", split_idx , "at: ", time.time())
        last_train = time.time()
        train(epoch)
        print("Last train takes: {} seconds".format(time.time()-last_train))

      else:
        train(epoch)
      scheduler.step()
    q.put(-1) 
    q.put(-1)
    q.put(False)   
    q.put("xxx")
    print("Waiting for subprocess to finish at: ", time.time())
    p.join()
    shm.close()
    shm.unlink()
except Exception as e:
  import traceback
  traceback.print_exc()
  print(f"Exception: {e}")
  #for c in psutil.Process(os.getpid()).children(recursive=True):
  #    c.kill()


print("Total time for streaming and training: {}".format(time.time()-training_streaming_start_time))

gpu_periodic_stat_finish_time = time.time()
stop_thread = True
gpu_th.join()

print("Finish gpu thread time: {}".format(time.time()-gpu_periodic_stat_finish_time))

print("The whole process took {} seconds".format(time.time()-start_time))
print("The last epoch took {:.1f} seconds with QQLE split {}".format(time.time()-last_epoch_start, split_idx))
sys.stdout.flush()



