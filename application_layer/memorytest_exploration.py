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
from torch.profiler import profile, record_function, ProfilerActivity
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
#    memory_stats_d = torch.cuda.memory_stats(0)
#    active = memory_stats_d['active_bytes.all.peak'] / (1024**2)
#    inactive = memory_stats_d['inactive_split_bytes.all.peak'] / (1024**2)
#    inactive = memory_stats_d['inactive_split_bytes.all.peak'] / (1024**2)
#    reserved = memory_stats_d['reserved_bytes.all.current'] / (1024**2)
#    print("torch cuda active, inactive, reserved ", active, inactive, reserved)
#    print("torch cuda memory stat ", torch.cuda.memory_stats(0))
    print("nvidia-smi gpu utilization ", _get_gpu_stats(0)[0][1])
    print("torch cuda memory allocated ", torch.cuda.memory_allocated(0) / (1024 ** 2))
    print("torch cuda max memory allocated ", torch.cuda.max_memory_allocated(0) / (1024 ** 2))
    print("torch cuda memory reserved ", torch.cuda.memory_reserved(0) / (1024 ** 2))
    # print(torch.cuda.max_memory_reserved(0) / (1024 ** 2))
    torch.cuda.reset_peak_memory_stats(0)
    print()

def _get_intermediate_outputs_and_time(model, input):
    #returns an array with sizes of intermediate outputs (in KBs), assuming some input
    output,sizes, int_time  = model(input,0,150, need_time=True)		#the last parameter is any large number that is bigger than the number of layers
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

#os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
idx = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{idx}"
cuda0 = torch.device('cuda:0')

#batch_sizes = [1, 10, 100]
#batch_sizes = [1, 1000]
batch_sizes = [128]
#batch_sizes = [1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'


for batch_size in batch_sizes:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    #print(torch.cuda.memory_stats(0))

    print_stats("Beginning of exp")
    img_tensor = torch.rand((batch_size,3,224,224)).to(device)
    print_stats(f"After loading input to cuda, input batch size {batch_size}")

    #['resnet18', 'resnet50', 'vgg11','vgg19', 'alexnet', 'densenet121']
    num_classes = 1000
    #model_test = build_my_resnet('resnet18', num_classes)
    #model_test = build_my_resnet('resnet50', num_classes)
    model_test = build_my_vgg('vgg11', num_classes)
    #model_test = build_my_vgg('vgg19', num_classes)
    #model_test = build_my_alexnet(num_classes)
    #model_test = build_my_densenet('densenet121', num_classes)
    

    all_layers = []
    remove_sequential(model_test, all_layers)
    print(all_layers)
    print(len(all_layers))

    if torch.cuda.is_available():
        model_test.cuda()
    print_stats("After loading model to cuda ")

    input = torch.rand((1,3,224,224)).to(device)
    input_size = np.prod(np.array(input.size())) * 4 / 4.5
    print(model_test(input, start=0, end=100)[1][22])
    sizes, int_time = _get_intermediate_outputs_and_time(model_test, input)
    sizes = np.array(sizes) * 1024


    print()
    input_size = img_tensor.element_size() * img_tensor.nelement() / (1024**2)
    print("Input size: ", input_size)
    print("Max output: ", max(sizes/1024./1024.*batch_size))
    print("Total output: ", sum(sizes/1024./1024.*batch_size))
    mod_sizes = [np.prod(np.array(p.size())) for p in model_test.parameters()]
    model_size = np.sum(mod_sizes) * 4 / (1024 * 1024)
    print("Model size: ", model_size)
    print()
    


#    tb_sizes, tb_times = _get_intermediate_outputs_and_time(model_test, torch.rand((batch_size,3,224,244)).to(device))
#    print("IDXS sorted by sizes", np.array(sizes).argsort().tolist())
#    np.set_printoptions(suppress=True)
#    print(np.sort(np.array(sizes/1024./1024*batch_size)))
#
#    print("Sizes + Times: ")
#    for i in zip(np.array(tb_sizes)/1024., tb_times):
#        print(i)
#    print()
#
#    print("Sizes + Times: ")
#    for i in zip(sizes/1024./1024*batch_size,int_time):
#        print(i)
#    print()

#    split_freeze_idx = len(model_test.all_layers)
#    server, client, vanilla, model_size, begtosplit_mem = get_mem_consumption(model_test, input, sizes,
#                                                                              split_freeze_idx, split_freeze_idx,
#                                                                              batch_size, batch_size)
#    print("All freeze MEM CONSUMP: ", "Total server ", server, " total client ", client, " vanilla ", \
#          vanilla, " model size ", model_size, " beg to split ", begtosplit_mem)
#    print()
#
#    split_freeze_idx = 0
#    server, client, vanilla, model_size, begtosplit_mem = get_mem_consumption(model_test, input, sizes,
#                                                                              split_freeze_idx, split_freeze_idx,
#                                                                              batch_size, batch_size)
#    print("No freeze MEM CONSUMP: ", "Total server ", server, " total client ", client, " vanilla ", \
#          vanilla, " model size ", model_size, " beg to split ", begtosplit_mem)
#    print()
#
#

    # just to put stats at 0 again after some checks
    print_stats("Before inferences")


    model_test.eval()
    with torch.no_grad():
        #for i in range(1,len(sizes)):
        for i in range(1,len(sizes)):
            #with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
            _ = model_test(img_tensor,0,i)
            print("!!!! Till layer ", i)
            #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            print_stats("After inference eval mode layer ")



    #model_test.eval()
    #with torch.no_grad():
    #    output, res = model_test(img_tensor,0,100)
    #print_stats("After inference eval mode")


    #with torch.inference_mode():
    #    output, res = model_test(img_tensor,0,100)
    #print_stats("After inference inference mode")

    model_test.train()
    _ = model_test(img_tensor,0,100)
    print_stats("After inference train mode")
    

    #for param in model_test.features.parameters():
    for param in model_test.parameters():
        param.requires_grad = False
    _, _ = model_test(img_tensor,0,100)
    print_stats("After inference train mode requires grad false")


    for param in model_test.parameters():
        param.requires_grad = True
    model_test.train()
    res, output = model_test(img_tensor,0,100)
    torch.optim.Adam(model_test.parameters(), lr=1e-3, betas=(0.9, 0.99))
    target = torch.rand(output.size()).to(device)
    print("TARGET: ", target.element_size() * target.nelement()/(1024 * 1024))
    loss = torch.nn.MSELoss()(output, target)
    loss.requires_grad = True
    print("LOSS: ", loss.element_size() * loss.nelement()/(1024 * 1024))
    print(loss)
    loss.backward()
    print_stats("After inference train mode and backward call")

    del output
    del res
    del input
    del model_test

    model_test_torchvision = torchvision.models.alexnet(num_classes=num_classes)
    model_test_torchvision.cuda()
    model_test_torchvision.eval()
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU],
                profile_memory=True, record_shapes=True, with_modules=True) as prof:
            model_test_torchvision(img_tensor)
    print(prof.export_chrome_trace("./test.json"))
    print(prof.key_averages().table())
    print_stats("TORCHVISION ALEXNET ")
    del model_test_torchvision

    del img_tensor
    print()
