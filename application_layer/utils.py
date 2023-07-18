'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import socket
import json
import concurrent.futures
import struct
import time
from time import sleep
import math
import subprocess
import shutil
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
from torchvision.models.densenet import _DenseLayer, _Transition  # only for freeze_sequential of densenet
import psutil
import numpy as np
import torch
import torchvision
import io
from io import BytesIO
import zipfile
from PIL import Image
import pickle
import iperf3
import functools
import operator
from swiftclient.service import SwiftService, SwiftPostObject
from timm.models import vision_transformer
import random

try:
    from application_layer.models import *
    from application_layer.dataset_utils import *
    from application_layer.mnist_utils import *
except:
    from models import *
    from dataset_utils import *
    from mnist_utils import *

SERVER_BATCH = 16 #250#16#128#256#128#25
NB_GPU = 2


#CACHED = True
#TRANSFORMED = True
#ALL_IN_COS = False#True


COMP_FILE_SIZE_DATASET = {
        'imagenet': 1000,
        #'imagenet': 128,
        'plantleave': 64,
        'inaturalist': 128
}

def get_model(model_str, dataset):
    """
    Returns a model of choice from the library, adapted also to the passed dataset
    :param model_str: the name of the required model
    :param dataset: the name of the dataset to be used
    :raises: ValueError
    :returns: Model object
    """
    num_class_dict = {'mnist': 10, 'cifar10': 10, 'cifar100': 100, 'imagenet': 1000, 'plantleave': 22, 'inaturalist': 8142}
    if dataset not in num_class_dict.keys():
        raise ValueError("Provided dataset ({}) is not known!".format(dataset))
    num_classes = num_class_dict[dataset]
    # Check if it is a native model or custom. The later should start with "my"
    if model_str.startswith('my'):
        # currently, we support: build_my_vgg, build_my_alexnet, build_my_inception, build_my_resnet, build_my_densenet
        model_str = model_str[2:]  # remove the "my" prefix
        if model_str.startswith('alex'):
            model = build_my_alexnet(num_classes)
        elif model_str.startswith('inception'):
            model = build_my_inception(num_classes)
        elif model_str.startswith('res'):
            model = build_my_resnet(model_str, num_classes)
        elif model_str.startswith('vgg'):
            model = build_my_vgg(model_str, num_classes)
        elif model_str.startswith('dense'):
            model = build_my_densenet(model_str, num_classes)
        elif model_str.startswith('vit'):
            model = build_my_vit(num_classes)
        else:
            ValueError("Provided model ({}) is not known!".format(model_str))
    else:
        models = {'convnet': Net,
                  'cifarnet': Cifarnet,
                  'cnn': CNNet,
                  'alexnet': torchvision.models.alexnet,
                  'resnet18': torchvision.models.resnet18,
                  'resnet34': torchvision.models.resnet34,
                  'resnet50': torchvision.models.resnet50,
                  'resnet152': torchvision.models.resnet152,
                  'inception': torchvision.models.inception_v3,
                  'vgg11': torchvision.models.vgg11,
                  'vgg16': torchvision.models.vgg16,
                  'vgg19': torchvision.models.vgg19,
                  'preactresnet18': PreActResNet18,
                  'googlenet': GoogLeNet,
                  'densenet121': torchvision.models.densenet121,
                  'densenet201': torchvision.models.densenet201,
                  'resnext29': ResNeXt29_2x64d,
                  'mobilenet': MobileNet,
                  'mobilenetv2': MobileNetV2,
                  'dpn92': DPN92,
                  'shufflenetg2': ShuffleNetG2,
                  'senet18': SENet18,
                  'efficientnetb0': EfficientNetB0,
                  'vit': vision_transformer.VisionTransformer,
                  'regnetx200': RegNetX_200MF}
        if model_str not in models.keys():
            raise ValueError("Provided model ({}) is not known!".format(model_str))
        model = models[model_str](num_classes=num_classes)
    return model


def _get_intermediate_outputs_and_time(model, input):
    print("In _get_intermediate_outputs_and_time")
    # returns an array with sizes of intermediate outputs (in KBs), assuming some input
    model.eval()
    with torch.no_grad():
        output, sizes, int_time, detailed_sizes, detailed_idx = model(input, 0, 150,
                                                                      need_time=True)  # the last parameter is any large number that is bigger than the number of layers
    model.train()
    return sizes.tolist(), int_time, detailed_sizes, detailed_idx  # note that returned sizes are of type Tensor


def _calculate_max(input_size, sizes):
    if len(sizes) > 0:
        sum_cons_sizes = [input_size]
        sum_cons_sizes.extend(sizes / 1024. / 1024.)
        sum_cons_sizes = [sum(sum_cons_sizes[i:i + 2]) for i in range(len(sum_cons_sizes) - 1)]
        max_output = max(sum_cons_sizes)
        return max_output, sum_cons_sizes
    return 0., sizes


# This function gets the memory consumption on both the client and the server sides with a given split_idx, freeze_idx,
# client batch size, server bach size, and model
# The function also returns the estimated memory consumption in the vanilla case
def get_mem_consumption(model_str, model_size, input_size, outputs_, split_idx, freeze_idx, client_batch, diff_bs1,
                        doutputs, detailed_idx, server_batch=SERVER_BATCH):
    if not (model_str == 'mydensenet121' or model_str == 'myresenet50'):
        diff_bs1 = 0.
    if freeze_idx < split_idx:  # we do not allow split after freeze index
        split_idx = freeze_idx
    dsplit = -1
    dsplit_server = -1
    dfreeze = -1
    for i, idx in enumerate(detailed_idx):
        if idx > (split_idx - 1) and dsplit_server == -1:
            dsplit_server = i - 1
        if idx > split_idx and dsplit == -1:
            dsplit = i - 1
        if idx > freeze_idx and dfreeze == -1:
            dfreeze = i - 1
    # outputs/=1024			#to make outputs also in KBs (P.S. it comes to here in Bytes)
    # input_size = np.prod(np.array(input.size()))*4/ (1024*1024)*server_batch
    begtosplit_sizes = doutputs[0:dsplit]
    intermediate_input_size = doutputs[dsplit_server] / (1024. * 1024.)
    splittofreeze_sizes = doutputs[dsplit:dfreeze]
    freezetoend_sizes = doutputs[dfreeze:]

    # note that before split, we do only forward pass so, we do not store gradients
    # after split index, we store gradients so we expect double the storage
    begtosplit_size, _ = _calculate_max(input_size, begtosplit_sizes)
    splittofreeze_size, _ = _calculate_max(intermediate_input_size, splittofreeze_sizes)
    freezetoend_size = np.sum(freezetoend_sizes) / (1024. * 1024.)

    # Just for debug
    print("Intermediate: ", intermediate_input_size)
    print(begtosplit_size)
    total_layers_size = np.sum(outputs_) / 1024. / 1024.
    print("Total layers size ", total_layers_size)

    # approximation with a small batch size on the sever
    total_server = (input_size + begtosplit_size + diff_bs1) * server_batch + model_size
    total_client = (
                               intermediate_input_size + splittofreeze_size + diff_bs1 + freezetoend_size * 2) * client_batch + model_size
    vanilla = (
                          input_size + begtosplit_size + splittofreeze_size + diff_bs1 + freezetoend_size * 2) * client_batch + model_size
    print("Server, client, server+client, vanilla ", total_server, total_client, total_server + total_client, vanilla)
    return total_server, total_client, vanilla, model_size, begtosplit_size, diff_bs1


def choose_split_idx(model_str, model, freeze_idx, client_batch, split_choice, split_idx_manual, device):
    GPU_in = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    if torch.cuda.is_available():
        model.to(device)
        _model_size = torch.cuda.max_memory_allocated(device) / (1024 ** 2) - GPU_in
    else:
        mod_sizes = [np.prod(np.array(p.size())) for p in model.parameters()]
        _model_size = np.sum(mod_sizes) * 4 / (1024 * 1024)
    print("IN SPLITTING ALGO")
    # a=torch.cuda.FloatTensor(1)
    # print("INIT ", _get_gpu_stats(0)[0][1], _get_gpu_stats(1)[0][1])
    # First of all, get the bandwidth
    client = iperf3.Client()
    client.duration = 1
    # client.server_hostname = "192.168.0.242"
    client.server_hostname = "192.168.0.246"
    #client.server_hostname = "192.168.0.189"
    while True:
        try:
            res = client.run()
            if res.error is None:
                bw = res.received_bps / 8  # bw now (after /8) is in Bytes/sec
                break
        except Exception as e:
            print(f"Error of iperf: {res.error}")
            print(e)
            sys.stdout.flush()
            res = None
            print("SLEEPING ")
            sleep(5)
            bw = 908.2855256674147 * 1024 * 1024 / 8
            break
    print(f"Recorded bandwidth: {bw * 8 / (1024 * 1024)} Mbps")
    # This function chooses the split index based on the intermediate output sizes and memory consumption
    input = torch.rand((1, 3, 224, 224)).to(device)
    # Step 1: select the layers whose outputs size is < input size && whose output < bw
    # input_size = np.prod(np.array(input.size()))*4/4.5		#I divide by 4.5 because the actual average Imagenet size is 4.5X less than the theoretical one
    input_size = input.element_size() * input.nelement() / (1024. ** 2)

    sizes, int_time, detailed_sizes, detailed_idx = _get_intermediate_outputs_and_time(model, input)
    sizes = np.array(sizes) * 1024  # sizes is in Bytes (after *1024)
    detailed_sizes = np.array(detailed_sizes) * 1024
    max_allocated_bs1 = torch.cuda.max_memory_allocated(0) / (1024. ** 2) - GPU_in
    max_output_bs1, sum_cons_sizes = _calculate_max(input_size, detailed_sizes)
    if np.argmax(sum_cons_sizes) != 0:
        diff_bs1 = max_allocated_bs1 - (input_size + _model_size + max_output_bs1)
    else:
        diff_bs1 = max_allocated_bs1 - (_model_size + max_output_bs1)
    del input
    print("Done intermediate outputs and time")

    #    print(f"Intermediate output sizes: {sizes*server_batch*100}")
    #    print(f"Min. of Input and BW {min(input_size*server_batch*100,bw)}")
    # note that input_size and sizes are both in Bytes
    # TODO: server_batch*100 depends on the current way of chunking and streaming data; this may be changed in the future
    print("Sizes ", sizes)
    print("Input_size ", input_size)
    print("TESTING *****************************")
    print("Input size, BW, MIN:")
    _input_size = np.prod(np.array(torch.rand((1,3,224,224)).size()))*4/4.5	
    print(_input_size * (1024. ** 2) * SERVER_BATCH * 100, bw, min(_input_size * (1024. ** 2) * SERVER_BATCH * 100, bw))
    #pot_idxs = np.where((sizes * SERVER_BATCH * 100 < min(_input_size * SERVER_BATCH * 100, bw)) & (sizes > 0))
    pot_idxs = np.where((sizes*client_batch < min(_input_size*client_batch, bw)) & (sizes > 0))
    # Step 2: select an index whose memory utilition is less than that in vanilla cases
    print("All candidates indexes: ", pot_idxs)
    print("SPLIT IDX CHOICE, split idx manual, freeze_idx: ", split_choice, split_idx_manual, freeze_idx)
    if split_choice == 'manual':
        split_idx = split_idx_manual
    elif split_choice == 'to_freeze':
        split_idx = freeze_idx
    elif split_choice == 'to_max':
        split_idx = np.argmax(sizes[:freeze_idx]) + 1
    else:
        split_idx = np.argmin(sizes[:freeze_idx]) + 1
    #    print(pot_idxs[0], bw, sizes*server_batch, input_size*server_batch)
    model_size, begtosplit_mem = 0, 0
    if split_choice == 'automatic':
        for idx in pot_idxs[0]:
            candidate_split = idx + 1  # to solve the off-by-one error
            if candidate_split > freeze_idx:
                break
            split_idx = candidate_split
            server, client, vanilla, model_size, begtosplit_mem, diff_bs1 = get_mem_consumption(model_str, _model_size,
                                                                                                input_size, sizes,
                                                                                                split_idx, freeze_idx,
                                                                                                client_batch, diff_bs1,
                                                                                                detailed_sizes,
                                                                                                detailed_idx)
            print("Candidate split ", candidate_split)
            print("Server, client, server+client, vanilla ", server, client, server + client, vanilla)
            print("Model size ", model_size)
            if server + client < vanilla:
                break
    if model_size == 0:
        _, _, _, _, begtosplit_mem, diff_bs1 = get_mem_consumption(model_str, _model_size, input_size, sizes, split_idx,
                                                                   freeze_idx, client_batch, diff_bs1, detailed_sizes,
                                                                   detailed_idx)
    # Note that, now I have all the pieces of memory consumption on the server:
    # input_size (in bytes), model_size (in MBs), begtosplit_mem (in MBs)
    # I group those in 2 categores: (a) not affected by the batch size (model size), and (b) scale with batch size (input and begtosplit)
    # Note the unification of units in the next line (all reported in MBs to be compatible with the output of nvidia-smi)
    # TODO change input sizm
    fixed, scale_with_bsz = model_size, input_size + begtosplit_mem + diff_bs1
    print("Fixed, scale_with_bsz ", fixed, scale_with_bsz)
    print("Mem usage ", _get_gpu_stats(0)[0][1], _get_gpu_stats(1)[0][1])
    return split_idx, (fixed, scale_with_bsz)


def get_mem_usage():
    # returns a dict with memory usage values (in GBs)
    mem_dict = psutil.virtual_memory()
    all_mem = np.array([v for v in mem_dict]) / (1024 * 1024 * 1024)  # convert to GB
    return {"available": all_mem[1], "used": all_mem[3],
            "free": all_mem[4], "active": all_mem[5],
            "buffers": all_mem[7], "cached": all_mem[8]}


# copied from https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/callbacks/gpu_stats_monitor.html#GPUStatsMonitor
import subprocess
import shutil
import threading

def _get_gpu_stats_other(gpu_id):
    """
    Run nvidia-smi to get the gpu stats
    """
    gpu_query = 'uuid,utilization.gpu,utilization.memory'
    format = 'csv,nounits,noheader'
    
    result = subprocess.run(
        #[shutil.which('nvidia-smi'), f'--query-gpu={gpu_query}', f'--format={format}', f'--id={gpu_id}', '-l 1'],
        [shutil.which('nvidia-smi'), f'--query-gpu={gpu_query}', f'--format={format}', f'--id={gpu_id}'],
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
        check=True,
        timeout=5
    )

    def _to_float(x: str) -> float:
        try:
            return float(x)
        except ValueError:
            return 0.

    stats = result.stdout.strip().split(os.linesep)
    #stats = [[_to_float(x) for x in s.split(', ')] for s in stats]
    return stats

def _get_gpu_stats(gpu_id):
    """Run nvidia-smi to get the gpu stats"""
    gpu_query = ",".join(["utilization.gpu", "memory.used", "memory.total"])
    format = 'csv,nounits,noheader'
    result = subprocess.run(
        [shutil.which('nvidia-smi'), f'--query-gpu={gpu_query}', f'--format={format}', f'--id={gpu_id}', '-l 1'],
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # for backward compatibility with python version 3.6
        check=True
    )

    def _to_float(x: str) -> float:
        try:
            return float(x)
        except ValueError:
            return 0.

    stats = result.stdout.strip().split(os.linesep)
    stats = [[_to_float(x) for x in s.split(', ')] for s in stats]
    return stats


def get_periodic_stat(stop):
    start_t = time.time()
    while True:
        if stop():
            break
        try:
            mem_free = []
            mem_used = []
            for i in range(NB_GPU):
                print('GPU ID: ', i, ' uuid,utilization.gpu,utilization.memory: ', _get_gpu_stats_other(i))

                res_i = _get_gpu_stats(i)[0]
                mem_free.append(res_i[2] - res_i[1])
                mem_used.append(res_i[1])
            print(f"Memory occupied: {mem_used} Time: {time.time() - start_t}", flush=True)
            time.sleep(1)
        except subprocess.TimeoutExpired:
            print("Timeout to get memory occupied")


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width = 100
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def prepare_transforms(dataset_name):
    if dataset_name.startswith('cifar'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return transform_train, transform_test
    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return transform, transform
    # TODO change
    elif dataset_name == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        return transform_train, transform_test
        # TODO transform not needed
        # return transforms.Compose([]), transforms.Compose([])

    elif dataset_name == 'plantleave':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        return transform_train, transform_test
    elif dataset_name == 'inaturalist':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        return transform_train, transform_test


def get_train_test_split(dataset_name, datadir, transform_train, transform_test):
    if dataset_name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=datadir, train=True, download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root=datadir, train=False, download=False, transform=transform_test)
    elif dataset_name == 'mnist':
        process_mnist(datadir)
        trainset = torchvision.datasets.MNIST(root=datadir, train=True, download=False, transform=transform_train)
        testset = torchvision.datasets.MNIST(root=datadir, train=False, download=False, transform=transform_test)
    elif dataset_name == 'imagenet':
        print("WARNING! Downloading the whole imagenet dataset is not recommended!")
        # First, we need to put images correctly in its folders
        os.system("cd {}; ./valprep.sh".format(os.path.join(datadir, 'val')))
        # Then, we load the Imagenet dataset
        trainset = torchvision.datasets.ImageFolder(root=datadir, transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=datadir, transform=transform_test)
    else:
        print("WARNING! Downloading the whole ", dataset_name, "dataset is not recommended!")
        return [], []
    return trainset, testset


def recvall(sock, n, rest=None):
    # Helper function to receive data from the server
    data = bytearray()
    if rest != None:
        data.extend(rest)
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if packet == bytearray(b'ERR'):
            os.kill(os.getpid(), 9)
            #raise Exception
        data.extend(packet)
        if data == bytearray(b'ERR'):
            os.kill(os.getpid(), 9)
            #raise Exception
    return data


def send_request(request_dict):
    # This function sends a request to the intermediate server through its socket and wait for the result
    # request_dict is the dict object that should be sent to the server
    HOST = '192.168.0.246'  # The server's hostname or IP address		#TODO: store this somewhere general later
    #HOST = '192.168.0.189'  # The server's hostname or IP address		#TODO: store this somewhere general later
    # HOST = '192.168.0.242'  # The server's hostname or IP address		#TODO: store this somewhere general later
    PORT = 65432  # The port used by the server
    # process request_dict
    options = request_dict['meta'].union(request_dict['header'])
    request_dict = {}
    for opt in options:
        contents = opt.split(":")
        request_dict[contents[0]] = contents[1]
    # Create a socket and send the required options
    #  print("Sending request.....")
    sys.stdout.flush()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(json.dumps(request_dict).encode('utf-8'))
        raw_msglen = recvall(s, 8)
        #    print("Raw message len: ", raw_msglen)
        try:
            msglen = struct.unpack('>Q', raw_msglen)[0]
            data = recvall(s, msglen)
        except:
            # send I size 4
            msglen = struct.unpack('>I', raw_msglen[:4])[0]
            rest = raw_msglen[4:]
            data = recvall(s, msglen, rest)

        print(f"Length of received data: {len(data)}")
        return data


def stream_batch(dataset_name, stream_dataset_len, swift, datadir, parent_dir, labels, transform, batch_size, lstart, lend, model,
                          mode='vanilla', split_idx=100, mem_cons=(0, 0), sequential=False, use_intermediate=False, CACHED=True, TRANSFORMED=True, ALL_IN_COS=False, NO_ADAPT=False):
    COMP_FILE_SIZE = COMP_FILE_SIZE_DATASET[dataset_name]
    stream_time = time.time()
    print("The mode is: ", mode)
    if mode == 'split':
        parallel_posts = int((lend - lstart) / COMP_FILE_SIZE)  # number of posts request to run in parallel
        post_step = int((lend - lstart) / parallel_posts)  # if the batch is smaller, it will be handled on the server
        lend = stream_dataset_len[dataset_name] if lend > stream_dataset_len[dataset_name] else lend
        print("Start {}, end {}, post_step {}\r\n".format(lstart, lend, post_step))
        post_objects = []
        images = []
        post_time = time.time()
        for i, s in enumerate(range(lstart, lend, post_step)):
            cur_end = s + post_step if s + post_step <= lend else lend
            cur_step = cur_end - s
            #print(cur_step)
            if NO_ADAPT:
                bs_server = cur_step
            else:
                bs_server = SERVER_BATCH
            opts = {"meta": {"Ml-Task:inference",
                             "dataset:" + dataset_name, "model:{}".format(model),
                             f"Batch-Size:{bs_server}",  # {}".format(int(cur_step//5)),
                             #f"Batch-Size:{cur_step}",  # {}".format(int(cur_step//5)),
                             "start:{}".format(s), "end:{}".format(cur_end),
                             #            "Batch-Size:{}".format(post_step),
                             #            "start:{}".format(lstart),"end:{}".format(lend),
                             "Split-Idx:{}".format(split_idx),
                             "Fixed-Mem:{}".format(mem_cons[0]),
                             "Scale-BSZ:{}".format(mem_cons[1]),
                             "COMP_FILE_SIZE:{}".format(COMP_FILE_SIZE)},
                    "header": {"Parent-Dir:{}".format(parent_dir)}}
            #          obj_name = "{}/ILSVRC2012_val_000".format(parent_dir)+((5-len(str(s+1)))*"0")+str(s+1)+".JPEG"
            if CACHED and TRANSFORMED:
                obj_name = f"{parent_dir}/vals{s}e{s + COMP_FILE_SIZE}.PTB.zip"
            else:
                obj_name = f"{parent_dir}/vals{s}e{s + COMP_FILE_SIZE}.zip"
            #      obj_name = "{}/ILSVRC2012_val_000".format(parent_dir)+((5-len(str(lstart+1)))*"0")+str(lstart+1)+".JPEG"
            post_objects.append(SwiftPostObject(obj_name, opts))  # Create multiple posts
        #      post_time = time.time()
        read_bytes = 0
        if use_intermediate:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(send_request, post_obj.options) for post_obj in post_objects]
                for fut in futures:
                    res = fut.result()
                    print("Future took {} seconds".format(time.time() - stream_time))
                    pickle_time = time.time()
                    read_bytes += int(len(res))
                    images.extend(pickle.loads(res))
                    print("Pickle took {} seconds".format(time.time() - pickle_time))
        else:
            for post_res in swift.post(container=dataset_name, objects=post_objects):
                if 'error' in post_res.keys():
                    print("error: {}, traceback: {}".format(post_res['error'], post_res['traceback']))
                read_bytes += int(len(post_res['result']))
                print("Executing one post (whose result is of len {} bytes) took {} seconds".format(
                    len(post_res['result']), time.time() - post_time))
                images.extend(pickle.loads(post_res['result']))  # images now should be a list of numpy arrays
                print("After deserialization, time is: {} seconds".format(time.time() - post_time))
                sys.stdout.flush()
        print("Read {} MBs for this batch".format(read_bytes / (1024 * 1024)))
        print("Executing all posts took {} seconds".format(time.time() - post_time))
        transform = None  # no transform required in this case

    else:  # mode=='vanilla'
        if not CACHED:
            objects = []
            num_objs = int((lend - lstart) / COMP_FILE_SIZE)
            step = int((lend - lstart) / num_objs)
            lend = stream_dataset_len[dataset_name] if lend > stream_dataset_len[dataset_name] else lend
            # prepare images that should be read
            for idx in range(lstart, lend, step):
                #      idstr = str(idx+1)
                #      obj_name = "{}/ILSVRC2012_val_000".format(parent_dir)+((5-len(idstr))*"0")+idstr+".JPEG"
                if idx+COMP_FILE_SIZE > lend:
                    obj_name = f"{parent_dir}/vals{idx}e{lend}.zip"
                else:
                    obj_name = f"{parent_dir}/vals{idx}e{idx + COMP_FILE_SIZE}.zip"

                objects.append(obj_name)
            opts = {
                'out_directory': os.path.join(os.environ['HOME'], "temp")}  # It does not matter....I have all the images anyway
            #    opts = {'out_file':'-'}
            # read all requested images
            images = []
            if sequential:
                queries = objects  # request them one by one then
            else:
                queries = swift.download(container=dataset_name, objects=objects, options=opts)
            read_bytes = 0
            infolists = []
            decompress_time = 0
            for query in queries:
                if sequential:
                    query = next(swift.download(container=dataset_name, objects=[query], options=opts))
                #      print(query)
                read_bytes += int(query['read_length'])
                #      print("Time till before unzipping {} seconds".format(time.time()-stream_time))
                #      with open(os.path.join(datadir, query['object']), 'rb') as f:
                #        image_bytes = f.read()
                #      img = np.array(Image.open(BytesIO(image_bytes)).convert('RGB'))
                #      images.append(img)
                # read the downloaded zip file (note that we use the out directory we passed up in opts)
                decompt = time.time()
                with open(os.path.join(os.environ['HOME'], "temp", query['object']), 'rb') as f:
                    bytes = f.read()
                #      zipbytes = query['contents']
                #      f2 = BytesIO(b''.join(zipbytes))
                #print("READ TEST :", len(bytes)/1024./1024.)
                f2 = BytesIO(bytes)
                zipff = zipfile.ZipFile(f2, 'r')  # as zipf:
                # infolist = zipf.infolist().copy()
                infolists.append(zipff)
                #              imgs = [np.array(Image.open(BytesIO(zipf.open(f3).read())).convert('RGB')) for f3 in infolist]
                #              for file in zipf.infolist():
                #                  with zipf.open(file) as f3:
                #                      img = np.array(Image.open(BytesIO(f3.read())).convert('RGB'))
                #                      images.append(img)
                #      print("Time after unzipping {} seconds".format(time.time()-stream_time))
                decompress_time += (time.time() - decompt)
            decompt = time.time()
            listToImages = lambda zipff: [np.array(Image.open(BytesIO(zipff.open(f3).read())).convert('RGB')) for f3 in
                                          zipff.infolist()]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                images = list(executor.map(listToImages, infolists))
            images = functools.reduce(operator.iconcat, images, [])
            print("Read {} MBs for this batch".format(read_bytes / (1024 * 1024)))
        elif CACHED:
            parallel_posts = int((lend - lstart) / COMP_FILE_SIZE)  # number of posts request to run in parallel
            post_step = int((lend - lstart) / parallel_posts)  # if the batch is smaller, it will be handled on the server
            lend = stream_dataset_len[dataset_name] if lend > stream_dataset_len[dataset_name] else lend
            print("Start {}, end {}, post_step {}\r\n".format(lstart, lend, post_step))
            post_objects = []
            images = []
            post_time = time.time()
            for i, s in enumerate(range(lstart, lend, post_step)):
                cur_end = s + post_step if s + post_step <= lend else lend
                cur_step = cur_end - s
                opts = {"meta": {"Ml-Task:inference",
                                 "dataset:" + dataset_name, "model:{}".format(model),
                                 f"Batch-Size:{COMP_FILE_SIZE}",  # {}".format(int(cur_step//5)),
                                 "start:{}".format(s), "end:{}".format(cur_end),
                                 #            "Batch-Size:{}".format(post_step),
                                 #            "start:{}".format(lstart),"end:{}".format(lend),
                                 "Split-Idx:{}".format(-1),
                                 "Fixed-Mem:{}".format(mem_cons[0]),
                                 "Scale-BSZ:{}".format(mem_cons[1]),
                                 "COMP_FILE_SIZE:{}".format(COMP_FILE_SIZE)},
                        "header": {"Parent-Dir:{}".format(parent_dir)}}
                #          obj_name = "{}/ILSVRC2012_val_000".format(parent_dir)+((5-len(str(s+1)))*"0")+str(s+1)+".JPEG"
                if not TRANSFORMED:
                    obj_name = f"{parent_dir}/vals{s}e{s + COMP_FILE_SIZE}.zip"
                else:
                    obj_name = f"{parent_dir}/vals{s}e{s + COMP_FILE_SIZE}.PTB.zip" 
                #      obj_name = "{}/ILSVRC2012_val_000".format(parent_dir)+((5-len(str(lstart+1)))*"0")+str(lstart+1)+".JPEG"
                post_objects.append(SwiftPostObject(obj_name, opts))  # Create multiple posts
            #      post_time = time.time()
            read_bytes = 0
            images = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(send_request, post_obj.options) for post_obj in post_objects]
                if not ALL_IN_COS:
                    for fut in futures:
                        res = fut.result()
                        read_bytes += int(len(res))
                        #images.extend(pickle.loads(res))
                        time_concurrent = time.time()
                        f2 = BytesIO(res)
                        zipff = zipfile.ZipFile(f2, 'r')
                        print("Decompress data took {} seconds".format(time.time() - time_concurrent))
                        if not TRANSFORMED:
                            images.extend(np.array(Image.open(io.BytesIO(zipff.open(f3).read())).convert('RGB')) for f3 in zipff.infolist())
                        else:
                            images.extend(np.array(torch.load(io.BytesIO(zipff.open(f3).read()))) for f3 in zipff.infolist())
                            transform=None
                        print("Total decompress data took {} seconds".format(time.time() - time_concurrent))
                else:
                    for fut in futures:
                        res = fut.result()
                        read_bytes += int(len(res))
                        images.extend(pickle.loads(res))
                        transform=None
            print("Read {} MBs for this batch".format(read_bytes / (1024 * 1024)))
    # use only labels corresponding to the required images
    labels = labels[lstart:lend]
    assert len(images) == len(labels)
    dataloader_time = time.time()
    imgs = np.array(images)
    dataset = InMemoryDataset(imgs, labels=labels, transform=transform, mode=mode, transformed=TRANSFORMED)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    print("Dataloader took {} seconds".format(time.time() - dataloader_time))
    print("Streaming {} data took {} seconds".format(dataset_name, time.time() - stream_time))
    #  decompress_time += (time.time()-decompt)
    #  print("Time taken for post processing the received compressed file: {} seconds".format(decompress_time))
    return dataloader

types = [torch.nn.modules.container.Sequential, _DenseLayer, _Transition, PatchEmbed]

globalFreezeIndex = 0


def freeze_sequential(network, all_layers):
    global globalFreezeIndex
    for layer in network.children():
        if type(layer) in types:
            freeze_sequential(layer, all_layers)
        else:
            all_layers.append(layer)
            if globalFreezeIndex >= len(all_layers):
                for param in layer.parameters():
                    param.requires_grad = False


def freeze_lower_layers(net, split_idx):
    # freeze the lower layers whose index < split_idx
    global globalFreezeIndex
    globalFreezeIndex = split_idx
    all_layers = []
    freeze_sequential(net, all_layers)
    # test if indeed frozen
#  total=0
#  frozen=0
#  for param in net.parameters():
#    if not param.requires_grad:
#      frozen+=1
#    total+=1
#  print("Total number of frozen layers: {} out of {} ".format(frozen,total))
