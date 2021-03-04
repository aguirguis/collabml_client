from swiftclient.service import SwiftService, SwiftPostObject, SwiftError
import pickle
import torchvision
from pytorch_cifar.utils import get_model

dataset = 'mnist'	#cifar10
model = 'convnet'	#resnet50
objects=['mnist/t10k-images-idx3-ubyte'] #['cifar-10-batches-py/data_batch_1']
swift = SwiftService()
step = 10000					#current limits: 9K with Cifarnet, 850 with ResNet50
task='inference'
for start in range(0,10000,step):
  end = start+step
  end = 10000 if end >= 10000 else end
  #ask to do inference for images [strat:end] from the test batch
  print("{} for data [{}:{}]".format(task,start,end))
  opts = {"meta": {"Ml-Task:{}".format(task),
	"dataset:{}".format(dataset),"model:{}".format(model),
        "Batch-Size:100","Num-Epochs:1",
        "Lossfn:cross-entropy","Optimizer:sgd",
	"start:{}".format(start),"end:{}".format(end)},
	"header": {"Parent-Dir:mnist"}}
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
