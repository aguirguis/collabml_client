#Upload dataset to swift DB...
#Based on: https://docs.openstack.org/python-swiftclient/latest/service-api.html#upload
#import swiftclient as client
from swiftclient.service import SwiftService, SwiftUploadObject, SwiftError
from swiftclient.exceptions import ClientException
import os
from os import walk
from os.path import join

def uploadf(swift, container_name, fname, dir):
  """
  add file or folder contents to swift db specific container
	Parameters:
	swift			Swift Service object to help upload the dataset
	container_name (str)	name of the container to which the filer or folder should be added
	fname (str)		the name of the file or folder that should be uploaded
	dir (str)		the directory of the file or the folder locally
  """
  objs = []
  for (_dir, _ds, _fs) in walk(join(dir,fname)):	#explore the directory
    if (_ds + _fs):	#found some files here
      objs.extend([join(_dir, _f) for _f in _fs])
  objs = [SwiftUploadObject(o, object_name=o[len(dir):]) for o in objs]		#strip the "dir" name from the object name
  for r in swift.upload(container_name, objs,options={'skip_identical': True}):		#checking if everything is Ok
    if r['success']:
      if 'object' in r:
        print(r['object']+ " uploaded!")
      elif 'for_object' in r:
        print('%s segment %s' % (r['for_object'], r['segment_index']))
    else:
      error = r['error']
      if r['action'] == "create_container":
        print('Warning: failed to create container '"'%s'%s", container_name, error)
      elif r['action'] == "upload_object":
        print("Failed to upload object %s to container %s: %s" %(container_name, r['object'], error))
      else:
        print("%s" % error)

def main():
  swift = SwiftService()
  dataset_name = 'imagenet'
  try:
    swift.stat(container=dataset_name)
    found = True
  except SwiftError as e:
    found = False
  if not found:
    swift.post(container=dataset_name)
  else:
    print("Container {} already exists".format(dataset_name))
  homedir = os.environ['HOME']
  fname = 'imagenet'
  uploadf(swift, dataset_name, fname, join(homedir,'dataset', fname))
  print('Uploaded {} successfully!'.format(fname))

if __name__ == "__main__":
  try:
    main()
  except ClientException as exc:
    print("error occured.", exc)
