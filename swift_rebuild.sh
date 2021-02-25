#!/bin/bash
swift-init all stop		#stop swift
cd ~/swift; sudo python3 setup.py develop		#rebuild swift
cd ~/swift; sudo chown -R ${USER}:${USER} swift.egg-info
startmain			#start swift
echo "" > ../swift_personal_log.txt	#clear logs
