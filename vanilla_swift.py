from swiftclient.service import SwiftService, SwiftPostObject, SwiftError, SwiftUploadObject
import pickle
import torchvision
from application_layer.utils import get_model
from time import time
import torch
import numpy as np

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
  if req == 'POST':
    post_requests(swift, objs)
  elif req == 'GET':
    get_requests(swift, objs)
  elif req == 'PUT':
    put_requests(swift, objs)

nums = np.arange(1, 10001, step=100)
reqs=['PUT', 'POST', 'GET']
res_dict = {}
for req in reqs:
  for num_req in nums:
    start_time = time()
    send_req(req, num_req)
    end_time = time()
    res_dict[req+"_"+str(num_req)] = end_time - start_time
    print("{} {} requests took {} seconds".format(num_req, req, end_time-start_time))
print(res_dict)
