#handy script to run experiments with main.py
import os
import numpy as np
curdir = os.getcwd()
execfile = os.path.join(curdir,'main.py')
#the following dict maps models to the split indeces we want to test
#these indeces were got manually with another scripts
model_dict={
	    'resnet18': [i for i in range(1,14)], #[1,2,3,4,5,6,7,8,9,10,11,12,13],
#	    'resnet50': [21], #[20,21],
#            'resnet152': [40,41,42,50,51,52,53,54,55], #[52,55],
#            'vgg16': [31,33],
#             'vgg11': [i for i in range(1,28)],
#            'vgg19': [34,35,36,37,38,39],
            'alexnet': [i for i in range(1,20)], #[15], #[9,15],
            'densenet121': [i for i in range(1,22)] #[19] #[14,19]
}
bw='UNLIMITED'		#This is the bandwidth used for testing....we use it here only to annotate the logFile
#logFile = 'splitExp_bw{}mbps'.format(bw)
logFile = 'sweepExpAllModels_bw{}GPUusagelog'.format(bw)
logFile_base = 'parallelBaselineAllModels_bw{}GPUusagelog'.format(bw)
for (model, split_idcs) in model_dict.items():
  freeze_idx = 11 if model == 'resnet18' else 19		#this should be constant all over the experiments
  for split_idx in split_idcs:
    #split execution
    os.system('python3 {} --dataset imagenet --model {} \
	 --num_epochs 1 --batch_size 1000 --split_idx {} --freeze_idx {} --freeze >> {}'.format(execfile,'my'+model,split_idx,freeze_idx,logFile))
    os.system('echo {} >> {}'.format('='*100,logFile))
    #app. layer execution parallel
  os.system('python3 {} --dataset imagenet --model {} \
       --num_epochs 1 --batch_size 1000 --freeze_idx {} --freeze >> {}'.format(execfile,model,freeze_idx,logFile_base))
  os.system('echo {} >> {}'.format('='*100,logFile_base))
#    app. layer execution sequential
#    os.system('python3 {} --dataset imagenet --model {} \
#         --num_epochs 1 --batch_size 1000 --split_idx {} --freeze --sequential >> {}'.format(execfile,model,split_idx,logFile))
#    os.system('echo {} >> {}'.format('='*100,logFile))
