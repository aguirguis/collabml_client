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

server = 'ssh root@192.168.0.246 ' #f'ssh -i {os.path.expanduser("~/.ssh/KeyPair-485c.pem")} root@121.37.173.24 -p 22 '
client = '' #f'ssh -i {os.path.expanduser("~/.ssh/KeyPair-485c.pem")} root@124.71.204.82 -p 22 '

def empty_gpu():
    torch.cuda.empty_cache()
    subprocess.Popen(shlex.split(client + f"pkill -f 'python3 {execfile}'"))
    time.sleep(10)

def start_server(vanilla, transformed, dataset, model, bsz, m_bw, CPU_, ba, iteration=0): 
    print("STARTING SERVER")
    if vanilla:
        if transformed:
            subprocess.Popen(shlex.split(server + f'"cd /root/swift/swift/proxy/mllib;\
                                     ST_AUTH_VERSION=1.0 ST_AUTH=http://192.168.0.246:8080/auth/v1.0     ST_USER=test:tester ST_KEY=testing \
                                     python3 server.py --cached --vanilla --transformed > test/multiple_ba_exp_{dataset}/server_{ba}vanilla_{model}_bs{bsz}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}_transformed_{iteration}" \
                                                     &'))
        else:
            subprocess.Popen(shlex.split(server + f'"cd /root/swift/swift/proxy/mllib;\
                                                                 ST_AUTH_VERSION=1.0 ST_AUTH=http://192.168.0.246:8080/auth/v1.0     ST_USER=test:tester ST_KEY=testing \
                                                                 python3 server.py --cached --vanilla > test/multiple_ba_exp_{dataset}/server_{ba}vanilla_{model}_bs{bsz}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}_{iteration}" \
                                                                                 &'))
    else:
        subprocess.Popen(shlex.split(server + f'"cd /root/swift/swift/proxy/mllib;\
                                                             ST_AUTH_VERSION=1.0 ST_AUTH=http://192.168.0.246:8080/auth/v1.0     ST_USER=test:tester ST_KEY=testing \
                                                             python3 server.py --cached --transformed > test/multiple_ba_exp_{dataset}/server_{ba}{model}_bs{bsz}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}_{iteration}" \
                                                                             &'))
    
    time.sleep(120)

def kill_server():
    print("KILLING SERVER")
    while True:
        subprocess.Popen(shlex.split(server + f'"kill -15 $(ps -A | grep python | awk \'{{print $1}}\')" \&'))
        p = subprocess.Popen(shlex.split(server + f'"ps -A | grep python "'), stdout=subprocess.PIPE)
        
        p_string = p.communicate()[0].decode("utf-8")

        if 'python' not in p_string:
            break
        else:
            for proc in p_string.splitlines():
                proc_name = proc.split()[0]
                #print(proc_name)
                subprocess.Popen(shlex.split(server + f'"kill -9 {proc_name}" \&'))
            time.sleep(10)



def run_model_exp_ba(batch_size, model, freeze_idx, ba, m_bw, CPU_=False, dataset='imagenet', vanilla=False, transformed=True, iteration = 0):
    os.system(client + f'python3 {emptycachefile}')
    
    if ba == 'no_adaptation_':
        opt_adapt = '--no_adaptation'
    else:
        opt_adapt = ''
    if vanilla:
        if transformed:
            #run vanilla
            os.system(client + f'python3 {execfile} --dataset {dataset} --model {model} --num_epochs 1 --batch_size {batch_size}\
                                                             --freeze --freeze_idx {freeze_idx} {"--cpuonly" if CPU_ else ""} --cached --transformed {opt_adapt}\
                                                              > {logdir}/multiple_ba_exp_{dataset}/{ba}vanilla_{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}_transformed_{iteration}')
        else:
            os.system(client + f'python3 {execfile} --dataset {dataset} --model {model} --num_epochs 1 --batch_size {batch_size}\
                                                             --freeze --freeze_idx {freeze_idx} {"--cpuonly" if CPU_ else ""} --cached {opt_adapt}\
                                                              > {logdir}/multiple_ba_exp_{dataset}/{ba}vanilla_{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}_{iteration}')
    else:
        err_code = os.system(client + f'python3 {execfile} --dataset {dataset} --model my{model} --num_epochs 1 --batch_size {batch_size}\
                                                         --freeze --freeze_idx {freeze_idx}  --use_intermediate {"--cpuonly" if CPU_ else ""} --cached --transformed {opt_adapt}')


        if err_code != 0:
            kill_server()
            start_server(vanilla, transformed, dataset, model, batch_size, m_bw, CPU_, ba, iteration)


        os.system(client + f'python3 {execfile} --dataset {dataset} --model my{model} --num_epochs 1 --batch_size {batch_size}\
                                             --freeze --freeze_idx {freeze_idx}  --use_intermediate {"--cpuonly" if CPU_ else ""} --cached --transformed {opt_adapt}\
                                              > {logdir}/multiple_ba_exp_{dataset}/{ba}{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}_{iteration}')


#models=['alexnet']
#freeze_idxs=[17]
dataset ='imagenet'


#models=['resnet18', 'resnet50', 'vgg11', 'vgg19', 'alexnet', 'densenet121', 'vit']
#freeze_idxs=[11, 21, 25, 36, 17, 20, 17]

models=['vgg19', 'densenet121', 'vit']
freeze_idxs=[36, 20, 17]


CPUs = [False]

#BSZs = [1000,2000,3000,4000,5000,6000,7000,8000]
#BSZs = [4000,5000,6000,7000,8000]
BSZs = [5000]
bw = 1024

cached = True
vanilla_b = [False]
#transformed_b = [True, False]


#ba = 'no_adaptation_'
#BAs = ['no_adaptation_', '']
BAs = ['']

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
                    for ba in BAs:
                        for it in range(10):
                            start_server(vanilla, transformed, dataset, model, bsz, m_bw, CPU_, ba, iteration=it)
                            print("STARTED SERVER ", model, str(bsz), ba)


                            print("START MODEL EXP")
                            run_model_exp_ba(bsz, model, freeze_idx, ba, m_bw, CPU_=CPU_, dataset=dataset, vanilla=vanilla,
                                           transformed=transformed, iteration=it)
                            print("FINISHED MODEL EXP")

                            kill_server()
                            print("KILLED SERVER")
