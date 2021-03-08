from swiftclient.service import SwiftService, SwiftPostObject, SwiftError
import pickle
import torchvision
from pytorch_cifar.utils import get_model
from time import time
import argparse

parser = argparse.ArgumentParser(description='Do ML computation in Swift')
parser.add_argument('--dataset', default='mnist', type=str, help='dataset to be used')
parser.add_argument('--model', default='convnet', type=str, help='model to be used')
parser.add_argument('--task', default='inference', type=str, help='ML task (inference or training)')
parser.add_argument('--batch_size', default=100, type=int, help='batch size for dataloader')
parser.add_argument('--num_epochs', default=10, type=int, help='number of epochs for training')
args = parser.parse_args()

dataset = args.dataset
model = args.model
task = args.task
batch_size = args.batch_size
num_epochs = args.num_epochs
print(args)

parent_dirs = {'imagenet':'imagenet', 'mnist':'mnist', 'cifar10':'cifar-10-batches-py'}
parent_dir = parent_dirs[dataset]
objs_invoke = {'imagenet':'imagenet/ILSVRC2012_val_00000001.JPEG',
		'mnist_training':'mnist/train-images-idx3-ubyte',
		'mnist_inference':'mnist/t10k-images-idx3-ubyte',
		'cifar10':'cifar-10-batches-py/test_batch'}
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
for start in range(0,dataset_size,step):
  end = start+step
  #ask to do inference for images [strat:end] from the test batch
  print("{} for data [{}:{}]".format(task,start,end))
  opts = {"meta": {"Ml-Task:{}".format(task),
	"dataset:{}".format(dataset),"model:{}".format(model),
        "Batch-Size:{}".format(batch_size),"Num-Epochs:{}".format(num_epochs),
        "Lossfn:cross-entropy","Optimizer:sgd",
	"start:{}".format(start),"end:{}".format(end)},
	"header": {"Parent-Dir:{}".format(parent_dir)}}
  post_objects=[SwiftPostObject(o,opts) for o in objects]
  for post_res in swift.post(
      container=dataset,
      objects=post_objects):
    if post_res['success']:
      print("Object '%s' POST success" % post_res['object'])
      body = post_res['result']
      if task == 'inference':
        inf_res = pickle.loads(body)
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
