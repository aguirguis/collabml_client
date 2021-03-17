from swiftclient.service import SwiftService, SwiftPostObject, SwiftError
import pickle
import torchvision
from pytorch_cifar.utils import get_model
from time import time
import torch
import argparse

parser = argparse.ArgumentParser(description='Do ML computation in Swift')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset to be used')
parser.add_argument('--model', default='convnet', type=str, help='model to be used')
parser.add_argument('--task', default='inference', type=str, help='ML task (inference or training)')
parser.add_argument('--batch_size', default=100, type=int, help='batch size for dataloader')
parser.add_argument('--num_epochs', default=10, type=int, help='number of epochs for training')
parser.add_argument('--split_idx', default=100, type=int, help='index at which computation is split between Swift and app. layer')
args = parser.parse_args()

dataset = args.dataset
model = args.model
task = args.task
batch_size = args.batch_size
num_epochs = args.num_epochs
split_idx = args.split_idx
print(args)

parent_dirs = {'imagenet':'val', 'mnist':'mnist', 'cifar10':'cifar-10-batches-py'}
parent_dir = parent_dirs[dataset]
objs_invoke = {'imagenet':'{}/ILSVRC2012_val_00000001.JPEG'.format(parent_dir),
		'mnist_training':'{}/train-images-idx3-ubyte'.format(parent_dir),
		'mnist_inference':'{}/t10k-images-idx3-ubyte'.format(parent_dir),
		'cifar10':'{}/test_batch'.format(parent_dir)}
try:
  obj = objs_invoke[dataset]
except:		#This should be mnist!
  obj = objs_invoke[dataset+"_"+task]
objects = [obj]
swift = SwiftService()
step = 10000
#If it is a training task, I do not want to invoke post multiple times
#If it is an inference task with a small dataset (mnist or cifar10), we can actually do it in one go
#In both cases, set the size to step so that for loop is entered once only
if task == 'training' or dataset == 'mnist' or dataset == 'cifar10':
  dataset_size = step
else:
  dataset_size = 50000		#Imagenet has 50K images for now
start_time = time()
#post_objects = []
for start in range(0,dataset_size,step):
  end = start+step
  #ask to do inference for images [strat:end] from the test batch
  print("{} for data [{}:{}]".format(task,start,end))
  opts = {"meta": {"Ml-Task:{}".format(task),
	"dataset:{}".format(dataset),"model:{}".format(model),
        "Batch-Size:{}".format(batch_size),"Num-Epochs:{}".format(num_epochs),
        "Lossfn:cross-entropy","Optimizer:sgd",
	"start:{}".format(start),"end:{}".format(end),
        "Split-Idx:{}".format(split_idx)
         },
	"header": {"Parent-Dir:{}".format(parent_dir)}}
#  obj_name =  "{}/ILSVRC2012_val_000".format(parent_dir)+((5-len(str(start+1)))*"0")+str(start+1)+".JPEG"
  post_objects = [SwiftPostObject(o,opts) for o in objects]
#print("Total number of objects to post: {}".format(len(post_objects)))
  for post_res in swift.post(
      container=dataset,
      objects=post_objects):
    if post_res['success']:
      print("Object '%s' POST success" % post_res['object'])
      print("Request took {} seconds".format(time()-start_time))
      body = post_res['result']
      if task == 'inference':
        inf_res = pickle.loads(body)
        if model.startswith("my"):	#new path for transfer learning toy example....this is not really inference
            model = get_model(model, dataset)		#use CPU only in this script...no need for GPU now
            final_res = []
            for int_res in inf_res:
                inputs = torch.from_numpy(int_res)
                logits = model(inputs, split_idx,100)		#continue the inference process here
                final_res.extend(logits.max(1)[1])
            print("Split inference results length: {}".format(len(final_res)))
        else:
            print("{} result length: {}".format(task, len(inf_res)))
      elif task == 'training':
        model_dict = pickle.loads(body)
        model = get_model(model, dataset)
        model.load_state_dict(model_dict)
    else:
      print("Object '%s' POST failed" % post_res['object'])

print("The whole process took {} seconds".format(time()-start_time))

#make sure metadata was posted successfully
#res = swift.download(container=dataset,objects=objects)
#next(res)
#for download_res in swift.download(container=dataset,objects=objects):
#  print(download_res)
