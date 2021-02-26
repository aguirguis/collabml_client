from swiftclient.service import SwiftService, SwiftPostObject, SwiftError

container = 'cifar10'
objects=['cifar-10-batches-py/readme.html','cifar-10-batches-py/test_batch']
swift = SwiftService()
opts = {"meta": {"Ml-task:inference","dataset:cifar10","model:cifarnet"}}
post_objects=[SwiftPostObject(o,opts) for o in objects]
for post_res in swift.post(
      container=container,
      objects=post_objects):
  if post_res['success']:
    print("Inference result: ", post_res['response_dict']['headers']['inf-res'])
    print("Object '%s' POST success" % post_res['object'])
  else:
    print("Object '%s' POST failed" % post_res['object'])

#make sure metadata was posted successfully
#res = swift.download(container=container,objects=objects)
#next(res)
#for download_res in swift.download(container=container,objects=objects):
#  print(download_res)
