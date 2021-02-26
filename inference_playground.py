from swiftclient.service import SwiftService, SwiftPostObject, SwiftError

container = 'cifar10'
objects=['cifar-10-batches-py/test_batch'] #'cifar-10-batches-py/data_batch_1']
swift = SwiftService()
step = 1000					#current limits: 9K with Cifarnet, 850 with ResNet50
for start in range(0,1000,step):
  end = start+step
  end = 10000 if end >= 10000 else end
  #ask to do inference for images [strat:end] from the test batch
  print("Inference for data [{}:{}]".format(start,end))
  opts = {"meta": {"Ml-task:inference",
	"dataset:cifar10","model:cifarnet",
	"start:{}".format(start),"end:{}".format(end)}}
  post_objects=[SwiftPostObject(o,opts) for o in objects]
  for post_res in swift.post(
      container=container,
      objects=post_objects):
    if post_res['success']:
      inf_res = str(post_res['result'])[1:-1].split()
      print("Inference result length: ", len(inf_res))
      print("Object '%s' POST success" % post_res['object'])
    else:
      print("Object '%s' POST failed" % post_res['object'])

#make sure metadata was posted successfully
#res = swift.download(container=container,objects=objects)
#next(res)
#for download_res in swift.download(container=container,objects=objects):
#  print(download_res)
