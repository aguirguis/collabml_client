from swiftclient.service import SwiftService, SwiftPostObject, SwiftError
import pickle
import torchvision
from pytorch_cifar.utils import get_model

dataset = 'imagenet' #'mnist'	#cifar10
model = 'resnet50' #'convnet'	#resnet50
task = 'training'	#'training'
parent_dir = 'imagenet' #'mnist' #'cifar-10-batches-py'
objects = ['imagenet/ILSVRC2012_val_00002456.JPEG']	#some image I am sure it's uploaded to Swift
#['mnist/train-images-idx3-ubyte'] #['mnist/t10k-images-idx3-ubyte'] #['cifar-10-batches-py/data_batch_1']
swift = SwiftService()
step = 10					#current limits: 9K with Cifarnet, 850 with ResNet50
for start in range(24440,24450,step):
  end = start+step
  #ask to do inference for images [strat:end] from the test batch
  print("{} for data [{}:{}]".format(task,start,end))
  opts = {"meta": {"Ml-Task:{}".format(task),
	"dataset:{}".format(dataset),"model:{}".format(model),
        "Batch-Size:1","Num-Epochs:1",
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

#make sure metadata was posted successfully
#res = swift.download(container=dataset,objects=objects)
#next(res)
#for download_res in swift.download(container=dataset,objects=objects):
#  print(download_res)
