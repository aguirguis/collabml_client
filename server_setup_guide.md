# Server Setup Guide 
Please follow this guide to set up the server side. The server hosts the cloud object store (i.e., OpenStack Swift).
The source of most of the commands presented in this guide is [here](https://docs.openstack.org/swift/latest/development_saio.html).
We use `Ubuntu 20.04.2 LTS` on the server.

## Setting up the object storage
1. Download the code:
```
cd $HOME
git clone https://github.com/aguirguis/swift.git
#checkout the ml-swift branch
cd swift; git checkout ml-swift; cd -
#For testing purposes, download the python-swiftclient custom code:
git clone https://github.com/aguirguis/python-swiftclient.git
cd python-swiftclient; git checkout ml-swift; cd -
#Although the client code is not used on the server (obviously), some individual scripts there are still useful
git clone https://github.com/aguirguis/collabml_client.git
```
2. Install the following prerequisites:
```
#system requirements
sudo apt-get update
sudo apt-get install python3.8
sudo apt-get install rsync
#dependencies
sudo apt-get install curl gcc memcached sqlite3 xfsprogs \
                     git-core libffi-dev python3-setuptools \
                     liberasurecode-dev libssl-dev axel
sudo apt-get install python3-coverage python3-dev python3-nose \
                     python3-xattr python3-eventlet \
                     python3-greenlet python3-pastedeploy \
                     python3-netifaces python3-pip python3-dnspython \
                     python3-mock
#python packages
cd $HOME/python-swiftclient; sudo pip3 install -r requirements.txt; sudo python3 setup.py develop; cd -
cd $HOME/swift; sudo pip3 install --no-binary cryptography -r requirements.txt; sudo python3 setup.py develop; cd -
```
3. Configure storage (we use the loopback device for storage):
```
sudo mkdir -p /srv
#In the following line, you should specify how big your disk is. 200GB is just an example.
sudo truncate -s 200GB /srv/swift-disk
sudo mkfs.xfs /srv/swift-disk
/srv/swift-disk /mnt/sdb1 xfs loop,noatime 0 0
sudo mkdir /mnt/sdb1
sudo mount -a
sudo mkdir /mnt/sdb1/1 /mnt/sdb1/2 /mnt/sdb1/3 /mnt/sdb1/4
sudo chown ${USER}:${USER} /mnt/sdb1/*
for x in {1..4}; do sudo ln -s /mnt/sdb1/$x /srv/$x; done
sudo mkdir -p /srv/1/node/sdb1 /srv/1/node/sdb5 \
              /srv/2/node/sdb2 /srv/2/node/sdb6 \
              /srv/3/node/sdb3 /srv/3/node/sdb7 \
              /srv/4/node/sdb4 /srv/4/node/sdb8
sudo mkdir -p /var/run/swift
sudo mkdir -p /var/cache/swift /var/cache/swift2 \
              /var/cache/swift3 /var/cache/swift4
sudo chown -R ${USER}:${USER} /var/run/swift
sudo chown -R ${USER}:${USER} /var/cache/swift*
# **Make sure to include the trailing slash after /srv/$x/**
for x in {1..4}; do sudo chown -R ${USER}:${USER} /srv/$x/; done
```
Then, add the following lines to `/etc/rc.local` (before the exit 0) - **Make sure to replace your-user-name:your-group-name with the correct values**:
```
mkdir -p /var/cache/swift /var/cache/swift2 /var/cache/swift3 /var/cache/swift4
chown <your-user-name>:<your-group-name> /var/cache/swift*
mkdir -p /var/run/swift
chown <your-user-name>:<your-group-name> /var/run/swift
```
4. Set up rsync and memcached:
```
sudo cp $HOME/swift/doc/saio/rsyncd.conf /etc/
sudo sed -i "s/<your-user-name>/${USER}/" /etc/rsyncd.conf
```
Then, edit the following line in **/etc/default/rsync**
```
RSYNC_ENABLE=true
```
Then, run
```
sudo setsebool -P rsync_full_access 1
#start rsync
sudo systemctl enable rsync
sudo systemctl start rsync
#verify rsync (12 lines should be printed with the next command)
rsync rsync://pub@localhost/
#start memcached
sudo systemctl enable memcached
sudo systemctl start memcached
```
5. Configure the Swift server:
```
sudo rm -rf /etc/swift
#populate the /etc/swift directory. Note that these files configure Swift with the same default configuration mentioned in the paper.
cd $HOME/swift/doc; sudo cp -r saio/swift /etc/swift; cd -
sudo chown -R ${USER}:${USER} /etc/swift
find /etc/swift/ -name \*.conf | xargs sudo sed -i "s/<your-user-name>/${USER}/"
```
**[Important]** Make sure that the IP in `\etc\swift\proxy-server.conf` is set to the IP of the server.

6. Set up useful scripts:
```
mkdir -p $HOME/bin
cd $HOME/swift/doc; cp saio/bin/* $HOME/bin; cd -
chmod +x $HOME/bin/*
#sample configuration to run tests:
cp $HOME/swift/test/sample.conf /etc/swift/test.conf
```
**[Important]** Make sure that the IP in `\etc\swift\test.conf` is set to the same IP in `\etc\swift\proxy-server.conf`.

7. Configure environment variables (replace {SERVER_IP} in the fifth command with the server IP, i.e., in `\etc\swift\proxy-server.conf`):
```
echo "export SWIFT_TEST_CONFIG_FILE=/etc/swift/test.conf" >> $HOME/.bashrc
echo "export PATH=${PATH}:$HOME/bin" >> $HOME/.bashrc
echo "export SAIO_BLOCK_DEVICE=/srv/swift-disk" >> $HOME/.bashrc
echo "export ST_AUTH_VERSION=1.0" >> $HOME/.bashrc
echo "export ST_AUTH=http://{SERVER_IP}:8080/auth/v1.0" >> $HOME/.bashrc
echo "export ST_USER=test:tester" >> $HOME/.bashrc
echo "export ST_KEY=testing" >> $HOME/.bashrc
```
8. Construct the rings and start the server:
```
remakerings
rebuildswift
#simple test to make sure everything is running
curl -v -H 'X-Storage-User: test:tester' -H 'X-Storage-Pass: testing' http://{SERVER_IP}:8080/auth/v1.0
swift stat
```

## Downloading the datasets
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
axel -n 30 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
tar xvf ILSVRC2012_img_val.tar; rm ILSVRC2012_img_val.tar; cd ..
#In case you will use the train and test datasets as well, you can download them as follows
axel -n 30 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar
tar xvf ILSVRC2012_img_test_v10102019.tar; rm ILSVRC2012_img_test_v10102019.tar
mkdir train; cd train;
axel -n 30 https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
tar xvf ILSVRC2012_img_train.tar; rm ILSVRC2012_img_train.tar
cd ~
```
The `valprep.sh` script can be found [here](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Uploading the dataset to the object storage
We prepared a script `collabml_client/dataset_uploader.py` that helps upload the datasets to the Swift object storage. Use it as follows:
```
cd $HOME/collabml_client; python3 dataset_uploader.py [mnist|cifar10|imagenet]; cd -
```
Note that if you choose to upload the individual images (rather than compressed images -- see below), you need to replace `compressed` with `val` in [this line](https://github.com/aguirguis/collabml_client/blob/main/dataset_uploader.py#L47).

Our experiments show that Swift works better with big objects. For this, we prepared a script `collabml_client/compress_imagenet.py` to compress multiple images into one object.
You should **do this step before uploading the dataset to the object storage**.
Update [this line](https://github.com/aguirguis/collabml_client/blob/main/compress_imagenet.py#L9) to specify how many images per object, and then run:
```
cd $HOME/collabml_client; python3 compress_imagenet.py; cd -
```
