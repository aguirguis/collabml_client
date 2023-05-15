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


def run_model_exp(batch_size, model, freeze_idx, CPU_=False, dataset='imagenet', vanilla=False, transformed=True):
    os.system(client + f'python3 {emptycachefile}')

    if vanilla:
        if transformed:
            #run vanilla
            os.system(client + f'python3 {execfile} --dataset {dataset} --model {model} --num_epochs 1 --batch_size {batch_size}\
                                                             --freeze --freeze_idx {freeze_idx} {"--cpuonly" if CPU_ else ""} --cached --transformed\
                                                              > {logdir}/reduction_exp_{dataset}/vanilla_{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}_transformed')
        else:
            os.system(client + f'python3 {execfile} --dataset {dataset} --model {model} --num_epochs 1 --batch_size {batch_size}\
                                                             --freeze --freeze_idx {freeze_idx} {"--cpuonly" if CPU_ else ""} --cached \
                                                              > {logdir}/reduction_exp_{dataset}/vanilla_{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}')
    else:
        os.system(client + f'python3 {execfile} --dataset {dataset} --model my{model} --num_epochs 1 --batch_size {batch_size}\
                                                         --freeze --freeze_idx {freeze_idx}  --use_intermediate {"--cpuonly" if CPU_ else ""} --cached --transformed')

        os.system(client + f'python3 {execfile} --dataset {dataset} --model my{model} --num_epochs 1 --batch_size {batch_size}\
                                             --freeze --freeze_idx {freeze_idx}  --use_intermediate {"--cpuonly" if CPU_ else ""} --cached --transformed\
                                              > {logdir}/reduction_exp_{dataset}/cached_test_split_{model}_bs{batch_size}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}')


models=['alexnet']
freeze_idxs=[17]
CPUs = [False]

dataset = 'imagenet'

BSZs = [1000,2000,3000,4000,5000,6000,7000,8000]
bw = 1024

cached = True
vanilla_b = [True, False]
#transformed_b = [True, False]

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
                    # VANILLA
                    # TRANSFORMED
                    # CACHED
                    if vanilla:
                        if transformed:
                            subprocess.Popen(shlex.split(server + f'"cd /root/swift/swift/proxy/mllib;\
                                                     ST_AUTH_VERSION=1.0 ST_AUTH=http://192.168.0.246:8080/auth/v1.0     ST_USER=test:tester ST_KEY=testing \
                                                     python3 server.py --cached --vanilla --transformed > test/reduction_exp_{dataset}/vanilla_{model}_bs{bsz}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}_transformed" \
                                                                     &'))
                        else:
                            subprocess.Popen(shlex.split(server + f'"cd /root/swift/swift/proxy/mllib;\
                                                                                 ST_AUTH_VERSION=1.0 ST_AUTH=http://192.168.0.246:8080/auth/v1.0     ST_USER=test:tester ST_KEY=testing \
                                                                                 python3 server.py --cached --vanilla > test/reduction_exp_{dataset}/vanilla_{model}_bs{bsz}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}" \
                                                                                                 &'))
                    else:
                        subprocess.Popen(shlex.split(server + f'"cd /root/swift/swift/proxy/mllib;\
                                                                             ST_AUTH_VERSION=1.0 ST_AUTH=http://192.168.0.246:8080/auth/v1.0     ST_USER=test:tester ST_KEY=testing \
                                                                             python3 server.py --cached --transformed > test/reduction_exp_{dataset}/cached_test_split_{model}_bs{bsz}_bw{m_bw}_{"cpu" if CPU_ else "gpu"}" \
                                                                                             &'))
                    time.sleep(120)

                    run_model_exp(bsz, model, freeze_idx, CPU_=CPU_, dataset=dataset, vanilla=vanilla,
                                   transformed=transformed)

                    while True:
                        subprocess.Popen(shlex.split(server + f'"kill -15 $(ps -A | grep python | awk \'{{print $1}}\')" \&'))
                        p = subprocess.Popen(shlex.split(server + f'"ps -A | grep python "'), stdout=subprocess.PIPE)
                        
                        if 'python' not in p.communicate()[0].decode("utf-8"):
                            break
                        else:
                            time.sleep(10)
