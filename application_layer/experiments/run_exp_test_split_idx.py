import os
import time
from time import sleep
#import pexpect
import subprocess
import shlex

homedir = os.path.expanduser("~")
projectdir = os.path.join(homedir, "collabml_client/application_layer")
execfile = os.path.join(projectdir,"main.py")
emptycachefile = os.path.join(projectdir,"experiments/emptycache.py")
logdir = os.path.join(projectdir,"logs")

#server = f'ssh -i {os.path.expanduser("~/.ssh/KeyPair-485c.pem")} root@121.37.173.24 -p 22 '
#client = f'ssh -i {os.path.expanduser("~/.ssh/KeyPair-485c.pem")} root@124.71.204.82 -p 22 '

server_adr = '192.168.0.239'

server = f'ssh root@{server_adr} ' #f'ssh -i {os.path.expanduser("~/.ssh/KeyPair-485c.pem")} root@121.37.173.24 -p 22 '
#server = 'ssh root@192.168.0.246 ' #f'ssh -i {os.path.expanduser("~/.ssh/KeyPair-485c.pem")} root@121.37.173.24 -p 22 '
client = '' #f'ssh -i {os.path.expanduser("~/.ssh/KeyPair-485c.pem")} root@124.71.204.82 -p 22 '

def empty_gpu():
    torch.cuda.empty_cache()
    subprocess.Popen(shlex.split(client + f"pkill -f 'python3 {execfile}'"))
    time.sleep(10)

def start_server(vanilla, transformed, dataset, model, bsz, m_bw, CPU_, idx): 
    print("STARTING SERVER")
    if vanilla:
        if transformed:
            subprocess.Popen(shlex.split(server + f'"cd /root/swift/swift/proxy/mllib;\
                                     ST_AUTH_VERSION=1.0 ST_AUTH=http://{server_adr}:8080/auth/v1.0     ST_USER=test:tester ST_KEY=testing \
                                     python3 server.py --cached --vanilla --transformed > test/idx_{dataset}/{idx}_server_vanilla_{model}_bs{bsz}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}_transformed" \
                                                     &'))
        else:
            subprocess.Popen(shlex.split(server + f'"cd /root/swift/swift/proxy/mllib;\
                                                                 ST_AUTH_VERSION=1.0 ST_AUTH=http://{server_adr}:8080/auth/v1.0     ST_USER=test:tester ST_KEY=testing \
                                                                 python3 server.py --cached --vanilla > test/idx_{dataset}/{idx}_server_vanilla_{model}_bs{bsz}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}" \
                                                                                 &'))
    else:
        subprocess.Popen(shlex.split(server + f'"cd /root/swift/swift/proxy/mllib;\
                                                             ST_AUTH_VERSION=1.0 ST_AUTH=http://{server_adr}:8080/auth/v1.0     ST_USER=test:tester ST_KEY=testing \
                                                             python3 server.py --cached --transformed > test/idx_{dataset}/{idx}_server_{model}_bs{bsz}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}" \
                                                                             &'))
    
    time.sleep(120)

def kill_server():
    print("KILLING SERVER")
    while True:
        subprocess.Popen(shlex.split(server + f'"kill -15 $(ps -A | grep python | awk \'{{print $1}}\')" \&'))
        p = subprocess.Popen(shlex.split(server + f'"ps -A | grep python "'), stdout=subprocess.PIPE)
        
        if 'python' not in p.communicate()[0].decode("utf-8"):
            break
        else:
            time.sleep(10)



def run_model_exp(batch_size, model, freeze_idx, idx, m_bw, CPU_=False, dataset='imagenet', vanilla=False, transformed=True):
    os.system(client + f'python3 {emptycachefile}')
    
    if vanilla:
        if transformed:
            #run vanilla
            os.system(client + f'python3 {execfile} --dataset {dataset} --model {model} --num_epochs 1 --batch_size {batch_size}\
                                                             --freeze --freeze_idx {freeze_idx} --split_choice manual --split_idx {idx} {"--cpuonly" if CPU_ else ""} --cached --transformed \
                                                              > {logdir}/idx_{dataset}/{idx}_vanilla_{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}_transformed')
        else:
            os.system(client + f'python3 {execfile} --dataset {dataset} --model {model} --num_epochs 1 --batch_size {batch_size}\
                                                             --freeze --freeze_idx {freeze_idx} --split_choice manual --split_idx {idx} {"--cpuonly" if CPU_ else ""} --cached \
                                                              > {logdir}/idx_{dataset}/{idx}_vanilla_{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}')
    else:
        err_code = os.system(client + f'python3 {execfile} --dataset {dataset} --model my{model} --num_epochs 1 --batch_size {batch_size}\
                                                         --freeze --freeze_idx {freeze_idx}  --use_intermediate --split_choice manual --split_idx {idx} {"--cpuonly" if CPU_ else ""} --cached --transformed')


        if err_code != 0:
            kill_server()
            start_server(vanilla, transformed, dataset, model, batch_size, m_bw, CPU_, idx)


        os.system(client + f'python3 {execfile} --dataset {dataset} --model my{model} --num_epochs 1 --batch_size {batch_size}\
                                             --freeze --freeze_idx {freeze_idx}  --use_intermediate --split_choice manual --split_idx {idx} {"--cpuonly" if CPU_ else ""} --cached --transformed \
                                              > {logdir}/idx_{dataset}/{idx}_{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}')


#models=['vgg11']
#freeze_idxs=[25]
dataset ='imagenet'


#models=['alexnet', 'resnet18', 'resnet50', 'vgg11', 'vgg19', 'densenet121', 'vit']
#freeze_idxs=[17, 11, 21, 25, 36, 20, 17]

#models=['resnet18', 'resnet50', 'vgg11', 'vgg19', 'densenet121', 'vit']
#freeze_idxs=[11, 21, 25, 36, 20, 17]

#models=['resnet50', 'vgg11', 'vgg19', 'alexnet', 'densenet121', 'vit']
#freeze_idxs=[21, 25, 36, 17, 20, 17]

#models = ['densenet121']#, 'resnet18']
#freeze_idxs = [20]#, 11]

#models=['densenet121', 'resnet18', 'alexnet', 'resnet50', 'vgg11', 'vgg19', 'vit']
#freeze_idxs=[20, 11, 17, 21, 25, 36, 17]

models=['vgg19', 'vit', 'densenet121', 'resnet18', 'alexnet', 'resnet50', 'vgg11']
freeze_idxs=[36, 17, 20, 11, 17, 21, 25]

#models=['vgg19']
#freeze_idxs=[36]


CPUs = [False]

#BSZs = [1000,2000,3000,4000,5000,6000,7000,8000]
#BSZs = [1000,2000,4000,6000,8000]
BSZs = [1000]
#BWs = [1024]
BWs = [12*1024, 1024]
#BWs = [12*1024]

cached = True
vanilla_b = [False]
#transformed_b = [True, False]


for bw in BWs:
    for bsz in BSZs:
        for CPU_ in CPUs:
            assert len(models) == len(freeze_idxs)
            wondershaper_exec = os.path.join(homedir, "wondershaper", "wondershaper")
            os.system(client + f'{wondershaper_exec} -c -a eth0')
            m_bw = bw * 1024
            os.system(client + f'{wondershaper_exec} -a eth0 -d {m_bw} -u {m_bw}')

            for vanilla in vanilla_b:
                if vanilla:
                    transformed_b = [True, False]
                else:
                    transformed_b = [True]

                for transformed in transformed_b:
                    for model, freeze_idx in zip(models, freeze_idxs):
                        for idx in range(freeze_idx, 0, -1):
                        #for idx in [8]:
                            start_server(vanilla, transformed, dataset, model, bsz, m_bw, CPU_, idx)
                            print("STARTED SERVER ", model, str(bsz), idx)

                            print("START MODEL EXP")
                            run_model_exp(bsz, model, freeze_idx, idx, m_bw, CPU_=CPU_, dataset=dataset, vanilla=vanilla,
                                           transformed=transformed)
                            print("FINISHED MODEL EXP")

                            kill_server()
                            print("KILLED SERVER")
