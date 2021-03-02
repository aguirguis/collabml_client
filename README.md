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
