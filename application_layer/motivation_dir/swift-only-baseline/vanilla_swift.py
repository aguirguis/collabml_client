from swiftclient.service import SwiftService, SwiftPostObject, SwiftError, SwiftUploadObject
from time import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from multiprocessing import Process

def ml_requests(swift, objs):
  #helper function to do ML requests (essentially inference)
  post_objects=[]
  for i,obj in enumerate(objs):
    opts = {"meta": {"Ml-Task:training",				#First choice: Do inference not training
      "dataset:imagenet","model:alexnet",			#Second choice: use Alexnet
      "Batch-Size:10", "Num-Epochs:1",						#Third choice: batch size of 1
      "Lossfn:cross-entropy","Optimizer:sgd",
      "start:%d" % i,"end:%d" % (i+1),				#Fourth choice: do inference for 1 images only
       },
      "header": {"Parent-Dir:{}".format('val')}}
    post_objects.append(SwiftPostObject(obj,opts))
  for post_res in swift.post(container='imagenet', objects=post_objects):
    if not post_res['success']:
      print(post_res)

def post_requests(swift, objs):
  #helper function to post objects
  opts = {"meta": {"comment:dummy"}}
  post_objects = [SwiftPostObject(o,opts) for o in objs]
  for post_res in swift.post(container='imagenet', objects=post_objects):
    if not post_res['success']:
      print("Failure")

def get_requests(swift, objs):
  #helper function to get objects
  opts = {'out_directory':'/root/dataset/imagenet'}       #so that we can have it directly in memory
  for get_res in swift.download(container='imagenet', objects=objs, options=opts):
    if not get_res['success']:
      print('Failure')

def put_requests(swift, objs):
  #helper function to put objects
  objs = ['/root/dataset/imagenet/'+obj for obj in objs]
  for put_res in swift.upload(container='imagenet', objects=objs):
    if not put_res['success']:
      print('Failure')

def send_req(req: str, num_req: int):
  #This function stress tests Swift
  #It sends "num_req" requests of type "req" to get how long does it take
  #PARAMs: req (type of the request), num_req (number of the requests to excute)
  #We fix the testing on Imagenet images
  obj_prefix = 'val/ILSVRC2012_val_000'	#to be followed by 00001.JPEG (for instance)
  objs = []
  #prepare number of objects to invoke equal to num_req
  for i in range(num_req):
    obj_name = obj_prefix+((5-len(str(i+1)))*"0")+str(i+1)+".JPEG"
    objs.append(obj_name)
  swift = SwiftService()
  #Check the type of the request
  start_time = time()
  num_proc = 10
  total = len(objs)
  if req == 'POST':
    if len(objs) >= num_proc:
      proc = []
      step=int(total/num_proc)
      for i in range(num_proc):
        start=i*step
        p = Process(target=post_requests, args=(swift, objs[start:start+step]))
        p.start()
        proc.append(p)
      for p in proc:
        p.join()
    else:
      post_requests(swift, objs)
  elif req == 'GET':
    get_requests(swift, objs)
  elif req == 'PUT':
    put_requests(swift, objs)
  elif req == 'ML':
    ml_requests(swift, objs)

nums = np.arange(1, 10001, step=100)
#reqs=['PUT', 'POST', 'GET', 'ML']
reqs=['ML']
res_dict = {}
for req in reqs:
  res = []
  for num_req in nums:
    start_time = time()
    send_req(req, num_req)
    end_time = time()
    res.append(end_time - start_time)
    print(res)
    sys.stdout.flush()
  res_dict[req] = res
  print(res)
  sys.stdout.flush()

print(res_dict)
#######Plotting the results
fontsize=35
fig = plt.gcf()
fig.set_size_inches(15, 8)
plt.subplots_adjust(top=0.95, bottom=0.15, right=0.95, left=0.13)
figs=[]
for req in res_dict:
  fig, = plt.plot(nums, res_dict[req], linewidth=5, label=req)
  figs.append(fig)
plt.legend(handles=figs, fontsize=fontsize)
plt.ylabel('Time (sec.)',fontsize=fontsize)
plt.xlabel('Number of requests',fontsize=fontsize)
plt.savefig('swift_performance.pdf')
