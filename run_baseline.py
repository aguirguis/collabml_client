#handy script to run experiments with swift-only baseline
import os
import numpy as np
curdir = os.getcwd()
execfile = os.path.join(curdir,'mlswift_playground.py')
models=['resnet18', 'resnet50', 'vgg11','vgg19', 'alexnet']
logFile_base = 'swiftOnlyBaseline'
for model in models:
  os.system('python3 {} --dataset imagenet --model {} --num_epochs 1 --batch_size 500 --task training >> {}'.format(execfile,model,logFile_base))
  os.system('echo {} >> {}'.format('='*100,logFile_base))
