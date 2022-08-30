import os
import numpy as np

import torch
import torch.nn as nn

import subprocess
import shutil

import torch
from torchvision.models import AlexNet
from torch import Tensor
import torch.nn as nn
from time import time
import sys

try:
    from application_layer.models import *
    from application_layer.dataset_utils import *
    from application_layer.mnist_utils import *
except:
    from models import *
    from dataset_utils import *
    from mnist_utils import *


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


# from gpu_mem_track import MemTracker
# gpu_t = MemTracker()
def print_stats(m):
    # gpu_t.track()
    print(m)
    print("nvidia-smi gpu utilization ", _get_gpu_stats(0)[0][1])
    print("torch cuda memory allocated ", torch.cuda.memory_allocated(0) / (1024 ** 2))
    print("torch cuda max memory allocated ", torch.cuda.max_memory_allocated(0) / (1024 ** 2))
    print("torch cuda memory reserved ", torch.cuda.memory_reserved(0) / (1024 ** 2))
    # print(torch.cuda.max_memory_reserved(0) / (1024 ** 2))
    torch.cuda.reset_peak_memory_stats(0)
    print()

def _get_intermediate_outputs_and_time(model, input):
    #returns an array with sizes of intermediate outputs (in KBs), assuming some input
    output,sizes, int_time = model(input,0,150, need_time=True)		#the last parameter is any large number that is bigger than the number of layers
    return sizes.tolist(), int_time

def get_mem_consumption(model, input, outputs, split_idx, freeze_idx, client_batch=1, server_batch=1):
    if freeze_idx < split_idx:  # we do not allow split after freeze index
        split_idx = freeze_idx
    outputs_ = outputs/1024  # to make outputs also in KBs (P.S. it comes to here in Bytes)
    input_size = np.prod(np.array(input.size())) * 4 / (1024 * 1024) * server_batch
    begtosplit_sizes = outputs_[0:split_idx]
    # intermediate_input_size = outputs[split_idx]/ (1024*1024)*client_batch
    intermediate_input_size = outputs_[split_idx - 1] / 1024 * client_batch
    splittofreeze_sizes = outputs_[split_idx:freeze_idx]
    freezetoend_sizes = outputs_[freeze_idx:]
    # Calculating the required sizes
    params = [param for param in model.parameters()]
    mod_sizes = [np.prod(np.array(p.size())) for p in params]
    model_size = np.sum(mod_sizes) * 4 / (1024 * 1024)
    # print("Model size ", model_size)
    # note that before split, we do only forward pass so, we do not store gradients
    # after split index, we store gradients so we expect double the storage
    begtosplit_size = np.sum(begtosplit_sizes) / 1024 * server_batch
    splittofreeze_size = np.sum(splittofreeze_sizes) / 1024 * client_batch
    freezetoend_size = np.sum(freezetoend_sizes) / 1024 * client_batch

    # Just for debug
    print("SPLIT INDEX: ", split_idx)
    print("INPUT SIZE: ", input_size)
    print("BEG TO SPLIT: ", begtosplit_size)
    print("SPLIT TO FREEZE: ", splittofreeze_size)
    print("FREEZE TO END: ", freezetoend_size)
    print("OUTPUT SIZES: ")
    sizes_print = outputs_/1024*server_batch
    for i, size in enumerate(sizes_print):
        print(i, size)
    print("MAX OUTPUT SIZE: ", np.max(sizes_print))
    print("MODEL SIZE: ", model_size)
    print("Intermediate: ", intermediate_input_size)
    #print("freezetoend_size ", freezetoend_size)
    total_layers_size = np.sum(outputs_) / 1024*server_batch
    print("Total layers size ", total_layers_size)

    total_server = input_size + model_size + begtosplit_size
    total_client = intermediate_input_size + model_size + splittofreeze_size + freezetoend_size * 2
    vanilla = input_size * (client_batch / server_batch) + model_size + \
              (begtosplit_size * (client_batch / server_batch)) + splittofreeze_size + freezetoend_size * 2
    return total_server, total_client, vanilla, model_size, begtosplit_size / server_batch

idx = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{idx}"
cuda0 = torch.device('cuda:0')

batch_sizes = [1, 10, 100, 1000, 5000]
#batch_sizes = [1, 1000]
#batch_sizes = [10]
device = 'cuda' if torch.cuda.is_available() else 'cpu'


for batch_size in batch_sizes:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    #print(torch.cuda.memory_stats(0))

    #print_stats("Beginning of exp")
    img_tensor = torch.rand((batch_size,3,224,224)).to(device)
    #print_stats(f"After loading input to cuda, input batch size {batch_size}")

    #['resnet18', 'resnet50', 'vgg11','vgg19', 'alexnet', 'densenet121']
    num_classes = 1000
    #model_test = build_my_resnet('resnet18', num_classes)
    #model_test = build_my_resnet('resnet50', num_classes)
    #model_test = build_my_vgg('vgg11', num_classes)
    #model_test = build_my_vgg('vgg19', num_classes)
    model_test = build_my_alexnet(num_classes)
    #model_test = build_my_densenet('densenet121', num_classes)

    if torch.cuda.is_available():
        model_test.cuda()
    #print_stats("After loading model to cuda ")

    s_time = time.time()
    model_test.eval()
    with torch.no_grad():
        output, res = model_test(img_tensor,0,100)
    print("TIME ", batch_size, ": ", time.time()-s_time)
    #print_stats("After inference eval mode")


#    with torch.inference_mode():
#        output, res = model_test(img_tensor,0,100)
#    print_stats("After inference inference mode")
#
#    model_test.train()
#    output, res = model_test(img_tensor,0,100)
#    print_stats("After inference train mode")
#
#
#    model_test.train()
#    output, res = model_test(img_tensor,0,100)
#    torch.optim.Adam(model_test.parameters(), lr=1e-3, betas=(0.9, 0.99))
#    target = torch.rand(output.size()).to(device)
#    print("TARGET: ", target.element_size() * target.nelement()/(1024 * 1024))
#    loss = torch.nn.MSELoss()(output, target)
#    print("LOSS: ", loss.element_size() * loss.nelement()/(1024 * 1024))
#    print(loss)
#    loss.backward()
#    print_stats("After inference train mode and backward call")

    del output
    del res
    del img_tensor
    del model_test

    print()