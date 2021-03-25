#handy script to run experiments with mlswift_playground.py
#I will use this basically to run training (with freezing lower layers) on Swift
import os
curdir = os.getcwd()
execfile = os.path.join(curdir,'mlswift_playground.py')
#the following dict maps models to the split indeces we want to test
#these indeces were got manually with another scripts
model_dict={
	    'resnet18': [11,13],
	    'resnet50': [20,21],
            'resnet152': [52,55],
            'vgg16': [31,33],
            'vgg19': [37,39],
            'alexnet': [9,15],
            'densenet121': [14,19]
}
logFile = 'trainOnSwiftWithFreeze'
for (model, split_idcs) in model_dict.items():
  for split_idx in split_idcs:
    #split execution
    os.system('python3 {} --dataset imagenet --model {} \
	 --num_epochs 1 --batch_size 1000 --split_idx {} --task training >> {}'.format(execfile,model,split_idx,logFile))
    os.system('echo {} >> {}'.format('='*100,logFile))
