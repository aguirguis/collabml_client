#handy script to run experiments with swift-only baseline
import os
import numpy as np
from multiprocessing import Process

def _run_swiftonly(process_idx, batch_size, num_processes, special_dir=""):
    curdir = os.getcwd()
    execfile = os.path.join(curdir,'mlswift_playground.py')
    models_dict={0:("myalexnet",17), 1:("myresnet18",11), 2:("myresnet50",21), 3:("myvgg11",25), 4:("myvgg19",36),5:("mydensenet121",20)}
    model, freeze = models_dict[0]		#for now, we only use Alexnet
    logFile_base = 'swiftOnlyBaseline' if special_dir == "" else special_dir
    os.system('python3 {} --dataset imagenet --model {} --num_epochs 1 --batch_size {} --task training\
	 --freeze_idx {} --gpu_id {} >> {}'.format(execfile,model,batch_size ,freeze, process_idx%2, logFile_base))
    os.system('echo {} >> {}'.format('='*100,logFile_base))

batch_size = 1000
num_tenants = 2
processes = []
for i in range(num_tenants):
    p = Process(target=_run_swiftonly, args=(i,batch_size,num_tenants, f"swiftOnlyRes_p{num_tenants}_trial2"))
    p.start()
    processes.append(p)
#Waiting from thee processes to finish
for p in processes:
    p.join()
