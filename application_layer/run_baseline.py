#!/usr/bin/env python3
#handy script to run experiments with main.py
import os
import numpy as np
import traceback



try:
    homedir = os.path.expanduser("~")
    projectdir = os.path.join(homedir, "collabml_client/application_layer")
    execfile = os.path.join(projectdir,"main.py")
    
    models=['resnet18', 'resnet50', 'resnet152', 'vgg11','vgg19', 'alexnet', 'densenet121', 'vit']
    bw='150'		#This is the bandwidth used for testing....we use it here only to annotate the logFile
    logFile_base = 'parallelBaselineMotivation_bw{}'.format(bw)

    wondershaper_exec = os.path.join(homedir,"wondershaper","wondershaper")
    os.system(f'{wondershaper_exec} -c -a eth0')
    m_bw = bw * 1024
    os.system(f'{wondershaper_exec} -a eth0 -d {m_bw} -u {m_bw}')

    for model in models:
        os.system('python3 {} --dataset imagenet --model {} --num_epochs 1 --batch_size 512 >> {}'.format(execfile,model,logFile_base))
        os.system('echo {} >> {}'.format('='*100,logFile_base))
        os.system('python3 {} --dataset imagenet --model {} --num_epochs 1 --batch_size 512 --cpuonly >> {}'.format(execfile,model,logFile_base+"CPU"))
        os.system('echo {} >> {}'.format('='*100,logFile_base+"CPU"))

    os.system(f'{wondershaper_exec} -c -a eth0')
    os.system(f'{wondershaper_exec} -a eth0 -d {1024*1024} -u {1024*1024}')

except Exception:
    print("HERE")
    print(traceback.format_exc())
