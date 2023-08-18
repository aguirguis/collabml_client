import os
import time
from time import sleep
#import pexpect
import subprocess
import shlex
import sys

homedir = os.path.expanduser("~")
projectdir = os.path.join(homedir, "collabml_client/application_layer")
execfile = os.path.join(projectdir,"main.py")
emptycachefile = os.path.join(projectdir,"experiments/emptycache.py")
logdir = os.path.join(projectdir,"florin-res")

server_adr = '192.168.0.239'

server = f'ssh root@{server_adr} ' 
client = '' 

def empty_gpu():
    torch.cuda.empty_cache()
    subprocess.Popen(shlex.split(client + f"pkill -f 'python3 {execfile}'"))
    time.sleep(10)

def start_server(vanilla, transformed, dataset, model, bsz, m_bw, CPU_, idx): 
    print("STARTING SERVER with paremeters", vanilla, transformed, dataset, model, str(bsz), str(m_bw), str(CPU_), str(idx))
    if vanilla:
        if transformed: #baseline 2, vanilla = no DNN exec, transformed = do pre-processing offline
            subprocess.Popen(shlex.split(server + f'"cd /root/swift/swift/proxy/mllib;\
                                     ST_AUTH_VERSION=1.0 ST_AUTH=http://{server_adr}:8080/auth/v1.0     ST_USER=test:tester ST_KEY=testing \
                                     python3 server.py --cached --vanilla --transformed > florin-res/idx_{dataset}/{idx}_server_vanilla_{model}_bs{bsz}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}_transformed.log" \
                                                     &'))
        else: #baseline 1, vanilla - no DNN exec, pre-processing done on client
            subprocess.Popen(shlex.split(server + f'"cd /root/swift/swift/proxy/mllib;\
                                                                 ST_AUTH_VERSION=1.0 ST_AUTH=http://{server_adr}:8080/auth/v1.0     ST_USER=test:tester ST_KEY=testing \
                                                                 python3 server.py --cached --vanilla > florin-res/idx_{dataset}/{idx}_server_vanilla_{model}_bs{bsz}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}.log" \
                                                                                 &'))
    else: #with split but only with offline pre-processing
        subprocess.Popen(shlex.split(server + f'"cd /root/swift/swift/proxy/mllib;\
                                                             ST_AUTH_VERSION=1.0 ST_AUTH=http://{server_adr}:8080/auth/v1.0     ST_USER=test:tester ST_KEY=testing \
                                                             python3 server.py --cached --transformed > florin-res/idx_{dataset}/{idx}_server_{model}_bs{bsz}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}.log" \
                                                                             &'))
    for i in range(120, 0, -1):
        sys.stdout.write(str(i)+' ')
        sys.stdout.flush()
        time.sleep(1) 
    print("\nSTARTED SERVER with paremeters", vanilla, transformed, dataset, model, str(bsz), str(m_bw), str(CPU_), str(idx))

def kill_server():
    print("KILLING SERVER")
    while True:
        subprocess.Popen(shlex.split(server + f'"kill -15 $(ps -A | grep server.py | awk \'{{print $1}}\')" \&'))
        p = subprocess.Popen(shlex.split(server + f'"ps -A | grep python "'), stdout=subprocess.PIPE)
        
        if 'python' not in p.communicate()[0].decode("utf-8"):
            break
        else:
            time.sleep(10)
    print("KILLED SERVER")



def run_model_exp(batch_size, model, freeze_idx, idx, m_bw, CPU_=False, dataset='imagenet', vanilla=False, transformed=True):
    os.system(client + f'python3 {emptycachefile}')
    print("\n\n\nSTARTING EXPERIMENT with paremeters", str(batch_size), model, str(freeze_idx), str(idx), str(m_bw), CPU_, dataset, vanilla, transformed)
    
    if vanilla:
        if transformed: 
            print ("Running BT")
            os.system(client + f'python3 {execfile} --dataset {dataset} --model {model} --num_epochs 1 --batch_size {batch_size}\
                                                             --freeze --freeze_idx {freeze_idx} --split_choice manual --split_idx {idx} {"--cpuonly" if CPU_ else ""} --cached --transformed \
                                                              > {logdir}/idx_{dataset}/{idx}_vanilla_{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}_transformed.log')
        else: #vanilla img
            print ("Running BIMG")
            os.system(client + f'python3 {execfile} --dataset {dataset} --model {model} --num_epochs 1 --batch_size {batch_size}\
                                                             --freeze --freeze_idx {freeze_idx} --split_choice manual --split_idx {idx} {"--cpuonly" if CPU_ else ""} --cached \
                                                              > {logdir}/idx_{dataset}/{idx}_vanilla_{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}.log')
    else:
        print ("Running HAPI")
        err_code = os.system(client + f'python3 {execfile} --dataset {dataset} --model my{model} --num_epochs 1 --batch_size {batch_size}\
                                                         --freeze --freeze_idx {freeze_idx} --split_choice manual --split_idx {idx} {"--cpuonly" if CPU_ else ""} --cached --transformed')
        #warmup run?

        if err_code != 0:
            kill_server()
            start_server(vanilla, transformed, dataset, model, batch_size, m_bw, CPU_, idx)

        #the model name starts with my
        os.system(client + f'python3 {execfile} --dataset {dataset} --model my{model} --num_epochs 1 --batch_size {batch_size}\
                                             --freeze --freeze_idx {freeze_idx}  --use_intermediate --split_choice manual --split_idx {idx} {"--cpuonly" if CPU_ else ""} --cached --transformed \
                                              > {logdir}/idx_{dataset}/{idx}_{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}.log')




#model_names=['vgg19', 'vit', 'densenet121', 'resnet18', 'alexnet', 'resnet50', 'vgg11']
#freeze_idxs=[36, 17, 20, 11, 17, 21, 25]
#models={k: v for k,v in zip(model_names,freeze_idxs)}
models={}
models['alexnet']=17
models['densenet121']=20
models['vgg11']=25
models['vgg19']=36
models['resnet18']=11
models['resnet50']=21
models['vit']=17

dataset ='imagenet'
CPUs = [False]
BSZs = [1000, 1250, 1500, 1750, 2000]
BWs = [12 * 1024]

cached = True
vanilla_b = [True]
#transformed_b = [True, False]

#other config in the code
# 1. storage batch size e.g. 16
# 2. nr of GPUs used      !! most important
# 3. request size e.g. 1000 - if you need smaller BS you need to reupload the data


kill_server()
for bw in BWs:
    for bsz in BSZs:
        for CPU_ in CPUs:
            #assert len(models) == len(freeze_idxs)
            wondershaper_exec = os.path.join(homedir, "wondershaper", "wondershaper")
            os.system(client + f'{wondershaper_exec} -c -a eth0')
            m_bw = bw * 1024
            os.system(client + f'{wondershaper_exec} -a eth0 -d {m_bw} -u {m_bw}')

            #for sol in "BT", "BIMG", "HAPI":
            for sol in "BT":
                if (sol == "BT"):
                    transformed, vanilla = [True, True]
                if (sol == "BIMG"):
                    transformed, vanilla = [False, True]
                if (sol == "HAPI"):
                    transformed, vanilla = [True, False]

                #for model in ['vgg19', 'vit', 'densenet121', 'resnet18', 'alexnet', 'resnet50', 'vgg11']:                
                start_server(vanilla, transformed, dataset, "xxxxx", bsz, m_bw, CPU_, 999)
                    
                #for model in ['vgg19', 'vit', 'densenet121', 'resnet18', 'alexnet', 'resnet50', 'vgg11']:                
                for model in ['alexnet']:                
                    freeze_idx=models[model]
                    #for idx in range(freeze_idx, 0, -1):
                    run_model_exp(bsz, model, freeze_idx, 999, m_bw, CPU_=CPU_, dataset=dataset, vanilla=vanilla, transformed=transformed)

                kill_server()
