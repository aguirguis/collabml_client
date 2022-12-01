Last login: Wed Nov 30 14:10:17 on ttys000
(base) pad@tsf-484-wpa-5-155 ~ % ssh -i .ssh/KeyPair-485c.pem root@121.37.173.24 -p 22
Welcome to Ubuntu 20.04.4 LTS (GNU/Linux 5.4.0-66-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

  System information as of Thu 01 Dec 2022 07:16:38 PM CST

  System load:  0.0                 Processes:             347
  Usage of /:   94.3% of 295.04GB   Users logged in:       0
  Memory usage: 9%                  IPv4 address for eth0: 192.168.0.246
  Swap usage:   0%

  => / is using 94.3% of 295.04GB

 * Strictly confined Kubernetes makes edge and IoT secure. Learn how MicroK8s
   just raised the bar for easy, resilient and secure K8s cluster deployment.

   https://ubuntu.com/engage/secure-kubernetes-at-the-edge

192 updates can be applied immediately.
122 of these updates are standard security updates.
To see these additional updates run: apt list --upgradable


	
	Welcome to Huawei Cloud Service

Last login: Wed Nov 30 21:10:28 2022 from 128.179.254.171
root@ecs-32b9-0001:~# client_loop: send disconnect: Broken pipe
(base) pad@tsf-484-wpa-5-155 ~ % ssh -i .ssh/KeyPair-485c.pem root@121.37.173.24 -p 22
Welcome to Ubuntu 20.04.4 LTS (GNU/Linux 5.4.0-66-generic x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage

  System information as of Thu 01 Dec 2022 09:23:06 PM CST

  System load:  0.09                Processes:             350
  Usage of /:   94.3% of 295.04GB   Users logged in:       1
  Memory usage: 9%                  IPv4 address for eth0: 192.168.0.246
  Swap usage:   0%

  => / is using 94.3% of 295.04GB

 * Strictly confined Kubernetes makes edge and IoT secure. Learn how MicroK8s
   just raised the bar for easy, resilient and secure K8s cluster deployment.

   https://ubuntu.com/engage/secure-kubernetes-at-the-edge

192 updates can be applied immediately.
122 of these updates are standard security updates.
To see these additional updates run: apt list --upgradable

New release '22.04.1 LTS' available.
Run 'do-release-upgrade' to upgrade to it.


	
	Welcome to Huawei Cloud Service

Last login: Thu Dec  1 19:16:40 2022 from 128.179.254.65
root@ecs-32b9-0001:~# cd swift/swift/proxy/mllib/
root@ecs-32b9-0001:~/swift/swift/proxy/mllib# vim server.py 
root@ecs-32b9-0001:~/swift/swift/proxy/mllib# vim server.py 
root@ecs-32b9-0001:~/swift/swift/proxy/mllib# vim ../../../../collabml_client/
.git/                             compress_imagenet.py              mlswift_playground.py             swiftOnlyBaseline
.gitignore                        compress_inaturalist.py           mymodels_playground/              test
.idea/                            compress_plantleave.py            parse_swiftOnly.py                venv/
README.md                         convert_images_to_tensor_file.py  requirements.txt                  
application_layer/                dataset_uploader.py               run_baseline.py                   
client_setup_guide.md             docker/                           server_setup_guide.md             
root@ecs-32b9-0001:~/swift/swift/proxy/mllib# vim ../../../../collabml_client/dataset_uploader.py 

    if (_ds + _fs):     #found some files here
      objs.extend([join(_dir, _f) for _f in _fs])
      print(objs)
  #objs = ['/root/dataset/imagenet/compressed/vals14000e15000.PTB.zip']
  objs_init = objs[:]

  while len(objs_init) != 0:
      objs = [SwiftUploadObject(o, object_name=o[len(dir):]) for o in objs_init]                #strip the "dir" name from the object name
      for r in swift.upload(container_name, objs):              #checking if everything is Ok
        if r['success']:
          if 'object' in r:
            print(r['object']+ " uploaded!")
            objs_init.remove(r['path'])
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
  parser = argparse.ArgumentParser(description='Swift Dataset Uploader')
  parser.add_argument('--dataset', type=str, help='dataset to be uploaded')
  args = parser.parse_args()
  dataset_name = args.dataset
  fnames = {'mnist':'mnist', 'cifar10':'cifar-10-batches-py', 'imagenet':'compressed', 'plantleave':'compressed', 'inaturalist':'compressed'}

  swift = SwiftService()
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
  fname = fnames[dataset_name]
  uploadf(swift, dataset_name, fname, join(homedir,'dataset',dataset_name))
  print('Uploaded {} successfully!'.format(fname))

if __name__ == "__main__":
  try:
    main()
  except ClientException as exc:
    print("error occured.", exc)
-- INSERT --                                                                                                                                        74,7          Bot

