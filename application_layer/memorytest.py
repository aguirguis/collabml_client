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

#os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
idx = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{idx}"
cuda0 = torch.device('cuda:0')

#batch_sizes = [1, 10, 100]
#batch_sizes = [1, 1000]
#batch_sizes = [10]
batch_sizes = [1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#models_name = ['resnet18', 'resnet50', 'vgg11','vgg19', 'alexnet', 'densenet121']
models_name = ['alexnet']

results = {}

for model_str in models_name:
#for batch_size in batch_sizes:
    results[model_str] = []
    batch_size = 8000
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    #print(torch.cuda.memory_stats(0))

    print()
    print()
    print(model_str)
    print_stats("Beginning of exp")

    #['resnet18', 'resnet50', 'vgg11','vgg19', 'alexnet', 'densenet121']
    num_classes = 1000
    if model_str.startswith('alex'):
        model_test = build_my_alexnet(num_classes)
    elif model_str.startswith('res'):
        model_test = build_my_resnet(model_str, num_classes)
    elif model_str.startswith('vgg'):
        model_test = build_my_vgg(model_str, num_classes)
    elif model_str.startswith('dense'):
        model_test = build_my_densenet(model_str, num_classes)
    elif model_str.startswith('vit'):
        model_test = build_my_vit(num_classes)
    
#    all_layers = []
#    remove_sequential(model_test, all_layers)
#    results[model_str].extend([len(all_layers)])
#    del all_layers
    
    #model_test = build_my_resnet('resnet18', num_classes)
    #model_test = build_my_resnet('resnet50', num_classes)
    #model_test = build_my_vgg('vgg11', num_classes)
    #model_test = build_my_vgg('vgg19', num_classes)
    #model_test = build_my_alexnet(num_classes)
    #model_test = build_my_densenet('densenet121', num_classes)

    if torch.cuda.is_available():
        model_test.cuda()
    model_size_cuda = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    print_stats("After loading model to cuda ")
    print("Model memory cuda ", model_size_cuda)
    mod_sizes = [np.prod(np.array(p.size())) for p in model_test.parameters()]
    model_size = np.sum(mod_sizes) * 4 / (1024 * 1024)
    print("Model size: ", model_size)


    input_vec = torch.rand((1,3,224,224)).to(device)
    #input_size_vec = np.prod(np.array(input_vec.size())) * 4 / 4.5
    input_size_vec = input_vec.element_size() * input_vec.nelement() / (1024**2)
    sizes, int_time = _get_intermediate_outputs_and_time(model_test, input_vec)
    sizes = np.array(sizes) * 1024.
    print_stats("Before inference with bs 1")
    model_test.eval()
    with torch.no_grad():
        model_test(input_vec,0,100)
    max_allocated_bs1 = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    sum_cons_sizes = [input_size_vec]
    sum_cons_sizes.extend(sizes/1024./1024.)
    sum_cons_sizes = [sum(sum_cons_sizes[i:i+2]) for i in range(len(sum_cons_sizes)-1)]
    max_output_bs1 = max(sum_cons_sizes)
    del input_vec

    img_tensor = torch.rand((batch_size,3,224,224)).to(device)
    print_stats(f"After loading input to cuda, input batch size {batch_size}")
    input_size_cuda = torch.cuda.max_memory_allocated(0) / (1024 ** 2) - model_size_cuda
    #input_size = img_tensor.element_size() * img_tensor.nelement() / (1024**2)
    print("Input size cuda: ", input_size_cuda)
    #print("Input size ", input_size)

    print()
    input_size = img_tensor.element_size() * img_tensor.nelement() / (1024**2)
    print("Input size: ", input_size)
    sum_cons_sizes = [input_size]
    sum_cons_sizes.extend(sizes/1024./1024.*batch_size)
    sum_cons_sizes = [sum(sum_cons_sizes[i:i+2]) for i in range(len(sum_cons_sizes)-1)]
    max_output = max(sum_cons_sizes)
    #max_output = max(sizes/1024./1024.*batch_size)
    sum_output = sum(sizes/1024./1024.*batch_size)
    print("Max output: ", max_output)
    print("Total output: ", sum(sizes/1024./1024.*batch_size))
    mod_sizes = [np.prod(np.array(p.size())) for p in model_test.parameters()]
    model_size = np.sum(mod_sizes) * 4 / (1024 * 1024)
    print("Model size: ", model_size)
    print(sizes/1024./1024.*batch_size)
    print()

    if np.argmax(sum_cons_sizes) != 0:
        #results[model_str].extend([input_size+sum_output+model_size, input_size+max_output+model_size])
        diff_bs1 = max_allocated_bs1 - (input_size_vec + model_size_cuda + max_output_bs1) 
#        print(input_size_vec, input_size)
#        print(max_output_bs1, max_output)
#        print("HERE")
        #results[model_str].extend([input_size+sum_output+model_size_cuda, input_size+max_output+model_size_cuda, diff_bs1*batch_size])
        results[model_str].extend([input_size+max_output+model_size_cuda, diff_bs1*batch_size])
    else:
        print("SPECIAL CASE ", model_str)
        #results[model_str].extend([input_size+sum_output+model_size, max_output+model_size]) 
        diff_bs1 = max_allocated_bs1 - (model_size_cuda + max_output_bs1) 
        #results[model_str].extend([input_size+sum_output+model_size_cuda, max_output+model_size_cuda, diff_bs1*batch_size])
        results[model_str].extend([max_output+model_size_cuda, diff_bs1*batch_size])


    # just to put stats at 0 again after some checks
    print_stats("Before inferences")


    model_test.eval()
    passed = True
    with torch.no_grad():
        try:
            model_test(img_tensor,0,100)
        except:
            passed = False
            print("WRONG VALUE ", model_str)
            pass
    
    max_allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
    print("MAX allocated ", max_allocated, max_allocated_bs1)
    results[model_str].extend([max_allocated])
    print_stats("After inference eval mode")
    
#    if passed:
#        del output
#        del res
#    del _
    del model_test
    del img_tensor
    print()


#    import gc
#    for obj in gc.get_objects():
#        try:
#            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#                print(type(obj), obj.size(), obj.get_device(), sys.getsizeof(obj.data))
#        except:
#            pass
    
#    print("TEST DIR")
#    for var_name, var_value in dict(locals()).items():
#        try:
#            #print(type(var_value))
#            #var_value.is_cuda
#            print(var_name, var_value)
#        except:
#            pass


print(results)
