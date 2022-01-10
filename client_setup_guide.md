# Client Setup Guide
Please follow this guide to set up the client side. We use `Ubuntu 20.04.2 LTS` on the client.



1. Download the code:
```
cd $HOME
git clone https://github.com/aguirguis/python-swiftclient.git
cd python-swiftclient; git checkout ml-swift; cd -
git clone https://github.com/aguirguis/collabml_client.git
#we use wondershaper to experiment with limited bandwidth
git clone https://github.com/magnific0/wondershaper.git
```
2. Install the following prerequisites:
```
sudo apt-get update
sudo apt-get install python3.8
#dependencies
sudo apt-get install curl gcc python3-setuptools
sudo apt-get install python3-coverage python3-dev python3-nose \
                     python3-pip
#python packages
cd $HOME/python-swiftclient; sudo pip3 install -r requirements.txt; sudo python3 setup.py develop; cd -
cd $HOME/collabml_client; sudo pip3 install -r requirements.txt; cd -
```
3. Configure environment variables (replace {SERVER_IP} in the second command with the server IP:
```
echo "export ST_AUTH_VERSION=1.0" >> $HOME/.bashrc
echo "export ST_AUTH=http://{SERVER_IP}:8080/auth/v1.0" >> $HOME/.bashrc
echo "export ST_USER=test:tester" >> $HOME/.bashrc
echo "export ST_KEY=testing" >> $HOME/.bashrc
#test that everything is Ok
swift stat
```
**Important**: make sure to update the server IP in [this line](https://github.com/aguirguis/collabml_client/blob/main/application_layer/utils.py#L404).
