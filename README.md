# swift_playground

Playing with OpenStack swift, the open-source object storage

## Instalation

1. Installed the requirements given [here](https://docs.openstack.org/swift/latest/getting_started.html)

2. Followed the guide [here](https://docs.openstack.org/swift/latest/development_saio.html) (using loopback device for storage)

## Notes

a. Used pip3 instead of pip and python3 instead of python all over the places.

b. In step 2 above, `sudo pip3 install -r requirements.txt` did not work. 
I ran it without `sudo` and then manually copied two packages to `/usr/local/lib....../dist-packages`

c. To rebuild the server, run

```
cd ~/swift; sudo python3 setup.py develop
cd ~/swift; sudo chown -R ${USER}:${USER} swift.egg-info
startmain
```

d. To stop the server run, `swift-init all stop`

e. How to get Auth_token? (I put the following exports to .bashrc on the development server for faster testing)

curl -v -H 'X-Storage-User: test:tester' -H 'X-Storage-Pass: testing' http://127.0.0.1:8080/auth/v1.0

You can set the following environment variables to access the storage directly 
(note that the service layer in the python client use these environment variables)

```
export ST_AUTH_VERSION=1.0
export ST_AUTH=http://10.221.117.112:8080/auth/v1.0
export ST_USER=test:tester
export ST_KEY=testing
```

f. CLI documentation [here](https://docs.openstack.org/python-swiftclient/latest/cli/index.html)

g. The client Service API documentation [here](https://docs.openstack.org/python-swiftclient/latest/service-api.html)

h. use `swift_rebuild.sh` to stop, revbuild, and restart the swift server after changing the source code

i. use `sudo python3 setup.py develop` to build the client from source too (make sure to checkout to `ml-swift` branch first)

j. Follow [this guide](https://help.atmail.com/hc/en-us/articles/201566464-Throttling-Bandwidth-using-Traffic-Controller-for-Linux) to test with custom BW throttling.

## Downloading datasets

`mkdir ~/dataset`

### Cifar10
```
cd ~/dataset
mkdir cifar10; cd cifar10
axel http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz
cd ~
```

### MNIST
```
cd ~/dataset
mkdir mnist; cd mnist
mkdir mnist; cd mnist		#This is not a typo; we do this so that we have the same structure as the other datasets
axel http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz; gzip train-images-idx3-ubyte.gz
axel http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz; gzip train-labels-idx1-ubyte.gz
axel http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz; gzip t10k-images-idx3-ubyte.gz
axel http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz; gzip t10k-labels-idx1-ubyte.gz
cd ~
```

### Imagenet

```
cd ~/dataset
mkdir imagenet; cd imagenet
wget http://www.image-net.org/challenges/LSVRC/2010/ILSVRC2010_test_ground_truth.txt
mkdir val; cd val;
cp ../ILSVRC2010_test_ground_truth.txt ./; mv ILSVRC2010_test_ground_truth.txt ILSVRC2012_validation_ground_truth.txt
axel -n 30 http://image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_val.tar
tar xvf ILSVRC2012_img_val.tar; rm ILSVRC2012_img_val.tar; cd ..
axel -n 30 http://image-net.org/challenges/LSVRC/2010/d5ef8751a0a1077596a929e9a224ee01/non-pub/ILSVRC2010_images_test.tar
tar xvf ILSVRC2010_images_test.tar; rm ILSVRC2010_images_test.tar
mkdir train; cd train;
axel -n 30 http://image-net.org/challenges/LSVRC/2010/d5ef8751a0a1077596a929e9a224ee01/non-pub/ILSVRC2010_images_train.tar
tar xvf ILSVRC2010_images_train.tar; rm ILSVRC2010_images_train.tar
cd ~
```
The `valprep.sh` script can be found [here](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
