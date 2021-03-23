#handy script to run experiments with main.py
import os
curdir = os.getcwd()
execfile = os.path.join(curdir,'main.py')
#the following dict maps models to the split indeces we want to test
#these indeces were got manually with another scripts
model_dict={
	    'resnet18': [11,13],
	    'resnet50': [21], #[20,21],
            'resnet152': [55], #[52,55],
            'vgg16': [31,33],
            'vgg19': [37,39],
            'alexnet': [15], #[9,15],
            'densenet121': [19] #[14,19]
}
bw=100		#This is the bandwidth used for testing....we use it here only to annotate the logFile
logFile = 'splitExp_bw{}mbps'.format(bw)
for (model, split_idcs) in model_dict.items():
  for split_idx in split_idcs:
    #split execution
    os.system('python3 {} --dataset imagenet --model {} \
	 --num_epochs 1 --batch_size 1000 --split_idx {} --freeze >> {}'.format(execfile,'my'+model,split_idx,logFile))
    os.system('echo {} >> {}'.format('='*100,logFile))
    #app. layer execution parallel
    os.system('python3 {} --dataset imagenet --model {} \
         --num_epochs 1 --batch_size 1000 --split_idx {} --freeze >> {}'.format(execfile,model,split_idx,logFile))
    os.system('echo {} >> {}'.format('='*100,logFile))
#    app. layer execution sequential
    os.system('python3 {} --dataset imagenet --model {} \
         --num_epochs 1 --batch_size 1000 --split_idx {} --freeze --sequential >> {}'.format(execfile,model,split_idx,logFile))
    os.system('echo {} >> {}'.format('='*100,logFile))
