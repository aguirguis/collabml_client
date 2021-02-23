#!/bin/bash
swift-init all stop
cd ~/swift; sudo python3 setup.py develop
cd ~/swift; sudo chown -R ${USER}:${USER} swift.egg-info
startmain
