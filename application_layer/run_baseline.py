#handy script to run experiments with main.py
import os
import numpy as np
curdir = os.getcwd()
execfile = os.path.join(curdir,'main.py')
models=['resnet18', 'resnet50', 'resnet152', 'vgg11','vgg19', 'alexnet', 'densenet121']
bw='UNLIMITED'		#This is the bandwidth used for testing....we use it here only to annotate the logFile
logFile_base = 'parallelBaselineMotivation_bw{}'.format(bw)
for model in models:
  os.system('python3 {} --dataset imagenet --model {} --num_epochs 1 --batch_size 500 >> {}'.format(execfile,model,logFile_base))
  os.system('echo {} >> {}'.format('='*100,logFile_base))
  os.system('python3 {} --dataset imagenet --model {} --num_epochs 1 --batch_size 500 --cpuonly >> {}'.format(execfile,model,logFile_base+"CPU"))
  os.system('echo {} >> {}'.format('='*100,logFile_base+"CPU"))
