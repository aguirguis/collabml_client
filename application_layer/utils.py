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
from multiprocessing import shared_memory

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

np.set_printoptions(linewidth=np.inf)
#CACHED = True
#TRANSFORMED = True
#ALL_IN_COS = False#True


COMP_FILE_SIZE_DATASET = {
        #'imagenet': 1000,
        'imagenet': 128,
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
        # add this to be faire because synchronize can slow down
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
            model = models[model_str](num_classes=num_classes)
    return model


def get_intermediate_outputs_and_time(model, input):
    print("In get_intermediate_outputs_and_time")
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
    #if not (model_str == 'mydensenet121' or model_str == 'myresenet50'):
    #    diff_bs1 = 0.

    #if (model_str == "densenet121"):
    #    diff_bs1 = 3.57
    #if (model_str == "resnet50"):
    #    diff_bs1 = 4.56


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

    #print("get_mem_consumption ", input_size, split_idx)
    print("get_mem_consumption doutputs ", doutputs)
    print("get_mem_consumption detailed_idx ", detailed_idx)
    print("get_mem_consumption splits ", dsplit, dsplit_server, dfreeze)

    # outputs/=1024			#to make outputs also in KBs (P.S. it comes to here in Bytes)
    # input_size = np.prod(np.array(input.size()))*4/ (1024*1024)*server_batch
    begtosplit_sizes = doutputs[0:dsplit]
    intermediate_input_size = doutputs[dsplit_server] / (1024. * 1024.)
    splittofreeze_sizes = doutputs[dsplit:dfreeze]
    freezetoend_sizes = doutputs[dfreeze:]

    vanilla_sizes = doutputs[0:dfreeze]

    # note that before split, we do only forward pass so, we do not store gradients
    # after split index, we store gradients so we expect double the storage
    begtosplit_size, _ = _calculate_max(input_size, begtosplit_sizes)
    splittofreeze_size, _ = _calculate_max(intermediate_input_size, splittofreeze_sizes)
    freezetoend_size = np.sum(freezetoend_sizes) / (1024. * 1024.)
    vanilla_size, _ = _calculate_max(input_size, vanilla_sizes)

    # Just for debug
    #print("get_mem_consumption intermediate_input_size: ", intermediate_input_size)
    print("get_mem_consumption begtosplit_size, splittofreeze_size, freezetoend_size, vanilla_size: ", begtosplit_size, splittofreeze_size, freezetoend_size , vanilla_size)
    total_layers_size = np.sum(outputs_) / 1024. / 1024.
    print("get_mem_consumption Total layers size ", total_layers_size)

    # approximation with a small batch size on the sever
    total_server = (input_size + begtosplit_size + diff_bs1) * server_batch + model_size
    total_client = (intermediate_input_size + splittofreeze_size + diff_bs1 + freezetoend_size * 2) * client_batch + model_size
    vanilla      = (input_size + vanilla_size + diff_bs1 + freezetoend_size * 2) * client_batch + model_size
    vanilla1      = (input_size + vanilla_size + freezetoend_size * 2) * client_batch + model_size

    #print("get_mem_consumption Server, client, server + client, vanilla, vanilla1 ", total_server, total_client, total_server + total_client, vanilla, vanilla1)
    return total_server, total_client, vanilla, vanilla1, begtosplit_size, intermediate_input_size



def estimate_memory(model_str, model, freeze_idx, client_batch, split_idx, device, _model_size):

    GPU_in = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    #print ("estimate_memory for ", split_idx, client_batch, " at: ", time.time(), " current max: ", GPU_in)
    #if torch.cuda.is_available():
    #    model.to(device)
    #    _model_size = torch.cuda.max_memory_allocated(device) / (1024 ** 2) - GPU_in
    #else:
    #    mod_sizes = [np.prod(np.array(p.size())) for p in model.parameters()]
    #    _model_size = np.sum(mod_sizes) * 4 / (1024 * 1024)
    #print("max_memory_allocated, max_memory_reserved, memory_allocated, max_memory_cached: ",torch.cuda.max_memory_allocated(device) / (1024 ** 2), torch.cuda.max_memory_reserved(device) / (1024 ** 2), torch.cuda.memory_allocated(device) / (1024 ** 2), torch.cuda.max_memory_cached(device) / (1024 ** 2))
    #print(torch.cuda.memory_stats(device="cuda"))
    #print(torch.cuda.memory_snapshot())

    input = torch.rand((1, 3, 224, 224)).to(device)
    # if input is an image
    # input_size = np.prod(np.array(input.size()))*4/4.5        #I divide by 4.5 because the actual average Imagenet size is 4.5X less than the theoretical one
    # if input is a tensor
    input_size = input.element_size() * input.nelement() / (1024. ** 2)

    sizes, int_time, detailed_sizes, detailed_idx = get_intermediate_outputs_and_time(model, input)    #here we do fwd pass as well
    #sizes - array layer output sizes
    #int_time - array forward pass per layer
    #detailed_sizes - same as sizes but some outputs are 0 if not change tensor (meaning no copy done for output since it's the same as input)
    #detailed_idx - array with indices appearing in detailed sizes e.g. [5,6,7..] if starting from 5
    print("estimate_memory sizes kB: ", sizes)
    print("estimate_memory detailed_sizes: ", detailed_sizes)
    print("estimate_memory int_time: ", int_time)
    print("estimate_memory detailed_idx: ", detailed_idx)
    
#    if ( "vit" in model_str):
 #           #detailed_sizes=[588, 0, 0, 0, 0, 5910, 5910, 5910, 5910, 5910, 5910, 5910, 5910, 5910, 5910, 5910, 5910, 3, 3, 3, 3.90625]
  #          detailed_sizes=[588, 0, 0, 0, 0, 4728, 4728, 4728, 4728, 4728, 4728, 4728, 4728, 4728, 4728, 4728, 4728, 3, 3, 3, 3.90625000448]

    #print("max_memory_allocated, max_memory_reserved, memory_allocated, max_memory_cached: ",torch.cuda.max_memory_allocated(device) / (1024 ** 2), torch.cuda.max_memory_reserved(device) / (1024 ** 2), torch.cuda.memory_allocated(device) / (1024 ** 2), torch.cuda.max_memory_cached(device) / (1024 ** 2))
    #print(torch.cuda.memory_stats(device="cuda"))
    #print(torch.cuda.memory_snapshot())
    GPU_after = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    sizes = np.array(sizes) * 1024  # sizes is in Bytes (after *1024)       
    detailed_sizes = np.array(detailed_sizes) * 1024
    max_allocated_bs1 = GPU_after - GPU_in
    max_output_bs1, sum_cons_sizes = _calculate_max(input_size, detailed_sizes)     #sum_cons_sizes - array with sum of two consecutive
    print("estimate_memory sum_consecutive_sizes: ", sum_cons_sizes)
    print("estimate_memory max_allocated_bs1, after_bs1, before_bs1 ", max_allocated_bs1, GPU_after, GPU_in)
    print("estimate_memory input_size: ", input_size)
    print("estcatimate_memory model_size: ", _model_size)
    print("estimate_memory max_output_bs1: ", max_output_bs1)

    #diff between real and estimation 
    if np.argmax(sum_cons_sizes) != 0: 
        diff_bs1 = max_allocated_bs1 - (input_size  + max_output_bs1)
        print("Calculate diff_bs1 as -- max_allocated_bs1 - (input_size  + max_output_bs1)", diff_bs1)       
    else:
        diff_bs1 = max_allocated_bs1 - (max_output_bs1)   
    del input

    if ("alexnet" in model_str):
        diff_bs1 = 0
    if ("resnet18" in model_str):
        diff_bs1 = 0
    if ("resnet50" in model_str):
        diff_bs1 = 4.56    
    if ("densenet" in model_str):
        diff_bs1 = 3.57
        #diff_bs1 = 1.75
    if ("vgg11" in model_str):
        diff_bs1 = 1.03
    if ("vgg19" in model_str):
        diff_bs1 = 1.5
    if ("vit" in model_str):
        diff_bs1 = 0

    #input_size (MB), sizes and detailed_sizes (bytes), model_size (MB)
    #for client_batch in [128,256,512,1024,1536,2048,3072]:

    oom_splits={}
    early_profiling_split=-1
    first_not_oom_split=-1
    for split_idx in range(1, freeze_idx+1):
        server, client, vanilla, vanilla1, begtosplit_mem, intermediate_input_size = get_mem_consumption(model_str, _model_size,
                                                                                            input_size, sizes,
                                                                                            split_idx, freeze_idx,
                                                                                            client_batch, diff_bs1,
                                                                                            detailed_sizes,
                                                                                            detailed_idx)
        print("Mem.est. BS/SIDX ",client_batch, split_idx, " ---  S/C/B+Diff/B ", server, client, vanilla, vanilla1, " --- ", _model_size, diff_bs1, input_size, intermediate_input_size)
        if (client >= 15109):  
            oom_splits[split_idx]=client
        else:
            if (first_not_oom_split == -1):  #assign just once
                if ("resnet18" in model_str and client_batch == 1536 and split_idx <= 3):
                    continue
                if ("resnet18" in model_str and client_batch == 2048 and split_idx <= 3):
                    continue       
                if ("resnet50" in model_str and client_batch == 1024 and split_idx <= 5):
                    continue    
                if ("densenet" in model_str and client_batch == 1024 and split_idx <= 3):
                    continue        
                if ("vgg19" in model_str and client_batch == 1536 and split_idx <= 18):
                    continue         
                if ("vit" in model_str and client_batch == 256 and split_idx <= 1):
                    continue
                if ("vit" in model_str and client_batch == 512 and split_idx <= 1):
                    continue 
                if ("vit" in model_str and client_batch == 1024 and split_idx <= 1):
                    continue
                first_not_oom_split = split_idx 

    #if (vanilla1 < 15109):
    if "alexnet" in model_str:
        early_profiling_split=0
    if "resnet18" in model_str and client_batch <= 1024:
        early_profiling_split=0
    if "resnet50" in model_str and client_batch <= 512:
        early_profiling_split=0 
    if "densenet" in model_str and client_batch <= 512:
        early_profiling_split=0
    if "vgg11" in model_str and client_batch <= 512:
        early_profiling_split=0
    if "vgg19" in model_str and client_batch <= 256: 
        early_profiling_split=0
    if "vit" in model_str and client_batch <= 128:
        early_profiling_split=0

    if (early_profiling_split == -1): #vanilla OOMs
        early_profiling_split = first_not_oom_split
    print("Profiling will occur with split: ", early_profiling_split)
    print("")
  

    return oom_splits, early_profiling_split


def get_nwbw_by_tcpdump():
    file1 = open('/root/tcpdump.out', 'r')
    Lines = file1.readlines()

    dict={}
    ctr=0
    avg=0
    for line in Lines:
        tokens = line.split()
        if (len(tokens) < 15):
            print("tcpdump truncated output. stop processing.")
            break
        if "epfl-florin3.65432" not in tokens[4] and "P" in tokens[6]: 
            #register the flow. This is entry 2/4 entries. Start of receiving data.
            #65432 > dport  [P]
            dport=tokens[4].split(".")[1][:-1]
            if dport not in dict:  #the FIN from the server can sometimes come first and the P flag is also there. Discard this, we already added the flow.
                dict[dport]=[tokens[0]]
        if "epfl-florin3.65432" in tokens[4] and "F" in tokens[6]:
            # > 65432 [F]
            dport=tokens[2].split(".")[1]
            if(len(dict[dport]) > 1):   #there could be repeated FINs, disregard them
                continue
            ctr=ctr+1
            ttt=dict[dport][0]
            dur=float(tokens[0])-float(ttt)
            dlen=int(tokens[10][:-1])
            dict[dport]=[ctr,dur,dlen, dlen/dur/1024/1024]
            print("tcpdump: ",ctr,dport,dur,dlen,dlen/dur/1024/1024, flush=True)
            avg=avg+dlen/dur/1024/1024
    print("tcpdump avg: ",ctr,avg/ctr)
    return 1024*1024*avg/ctr


def get_nwbw_by_tcpdump2(par):
    file1 = open('/root/tcpdump.out', 'r')
    Lines = file1.readlines()

    dict={}
    ctr=0
    avg=0

    par_st=[-1000]
    par_len=0
    par_avg=0
    par_avg_cnt=0

    for line in Lines:
        tokens = line.split()
        if (len(tokens) < 15):
            print("tcpdump truncated output. stop processing.")
            break
        if "epfl-florin3.65432" not in tokens[4] and "P" in tokens[6]:
            #register the flow. This is entry 2/4 entries. Start of receiving data.
            #65432 > dport  [P]
            dport=tokens[4].split(".")[1][:-1]
            if dport not in dict:  #the FIN from the server can sometimes come first and the P flag is also there. Discard this, we already added the flow.
                dict[dport]=[tokens[0]]
                par_st.append(float(tokens[0]))
        if "epfl-florin3.65432" in tokens[4] and "F" in tokens[6]:
            # > 65432 [F]
            dport=tokens[2].split(".")[1]
            if(len(dict[dport]) > 1):   #there could be repeated FINs, disregard them
                continue
            ctr=ctr+1
            ttt=dict[dport][0]
            dur=float(tokens[0])-float(ttt)
            dlen=int(tokens[10][:-1])
            dict[dport]=[ctr,dur,dlen, dlen/dur/1024/1024]
            print("tcpdump: ",ctr,dport,ttt,float(tokens[0]),dur,dlen,dlen/dur/1024/1024, flush=True)
            avg=avg + dlen/dur/1024/1024

            par_len = par_len + dlen
            if (ctr % par == 0):
                #print (tokens[0], par_st[ctr - par + 1])
                #print (par_st)
                par_dur=float(tokens[0])-par_st[ctr - par + 1]
                print("tcpdump par: ", int(ctr/par), par_dur, par_len, par_len/par_dur/1024/1024, flush=True)
                par_avg_cnt += 1
                par_avg += par_len/par_dur/1024/1024
                par_len=0

    print("tcpdump avg: ", ctr, avg/ctr)
    print("tcpdump par avg: ", par_avg_cnt, par_avg/par_avg_cnt)
    print("tcpdump par avg per req: ", par_avg_cnt, par_avg/par_avg_cnt/par)
    return 1024*1024*par_avg/par_avg_cnt/par




def _estimate_c_train_total(times_for_prediction1, times_for_prediction2, nb_inferences, idx_potential, output_size_layer, client_batch, early_profiling_split):
    nr_waves = math.ceil(client_batch/512)
    c_train_dataloader = output_size_layer*1.0190668e-10
    c_train_copy_to_gpu = output_size_layer*2.70339617e-10

    c_stream_pickle_load = output_size_layer * 4.76287825e-10    
    c_stream_dataloader = output_size_layer * 4.57389807e-10                                     
    
    #c_train_forward_pass = times_for_prediction2['inference_time'][idx_potential:].sum() * nr_waves

    #idx_potential runs for all splits (initially wout caring for OOM). For splits < early_profling_split, the sum
    #below is not correct, since the times from the server need to be multiplied by nr_waves and those for the client not.
    #however, splits < early_profiling_split will OOM and not be chosen so it doesn't matter if this calculation is not correct
    c_train_forward_pass = times_for_prediction2['inference_time'][idx_potential:].sum() 
    c_train_backward_pass = times_for_prediction1['c_train_backward_pass']

    c_train_total = c_train_dataloader+c_train_copy_to_gpu+c_train_forward_pass+c_train_backward_pass
    #c_train_total = c_train_dataloader+c_train_copy_to_gpu+c_train_forward_pass+c_train_backward_pass + c_stream_pickle_load + c_stream_dataloader

    #print("Estimated client times ", c_train_dataloader, c_train_copy_to_gpu, c_train_forward_pass, c_train_backward_pass, c_train_total)

    #print("Estimated CLI times XX ", idx_potential, round(c_train_dataloader,5), round(c_train_copy_to_gpu,5), round(c_train_forward_pass,5), round(c_train_backward_pass,5), round(c_train_total,5))
    print("Estimated CLI times XX ", idx_potential, round(c_stream_pickle_load,5), round(c_stream_dataloader,5), round(c_train_dataloader,5), round(c_train_copy_to_gpu,5), round(c_train_forward_pass,5), round(c_train_backward_pass,5), round(c_train_total,5))

    not_parallel = c_stream_pickle_load + c_stream_dataloader #+ 0.09
    #print("Estimated c_train_total ", c_train_total)

    return c_train_total, not_parallel


def oldcode(client_batch):
        #THIS IS IN BYTES PER SECOND (NOT !! bits)
    if (client_batch == 128):
        #bw = 1 * 1024 * 1024
        bw = 1050 * 1024 * 1024
    if (client_batch == 256):
        bw = 750 * 1024 * 1024
    if (client_batch == 512):
        bw = 500 * 1024 * 1024
    if (client_batch == 1024):
        bw = 550 * 1024 * 1024
    if (client_batch == 1536):
        bw = 600 * 1024 * 1024    
    if (client_batch == 2048):
        bw = 700 * 1024 * 1024    
    if (client_batch == 3072):
        bw = 800 * 1024 * 1024  

    if (client_batch >= 256):
        #bw =  0.65 * 1024 * 1024
        bw = 650 * 1024 * 1024    

    if (client_batch == 128):
        bw = 0.85 * nwbw * 1024 * 1024 / 8
    if (client_batch >= 256):
        bw = 0.65 * nwbw * 1024 * 1024 / 8


def _estimate_c_stream_total_iteration(times_for_prediction1, times_for_prediction2, nb_inferences, idx_potential, output_size_layer, bw, client_batch, nwbw):
    s_read_data_shm = times_for_prediction1['s_read_data_shm']
    s_read_model_shm_to_gpu = times_for_prediction1['s_read_model_shm_to_gpu']

    bw=nwbw

    #s_inf_to_pytorch = np.array(times_for_prediction['s_inf_to_pytorch'])
    #s_inf_copy_to_gpu = np.array(times_for_prediction['s_inf_copy_to_gpu'])
    #s_inf_to_numpy = np.array(times_for_prediction['s_inf_to_numpy'])

    s_inf_to_pytorch = times_for_prediction1['s_inf_to_pytorch']
    s_inf_copy_to_gpu = times_for_prediction1['s_inf_copy_to_gpu']
    s_inf_to_numpy = times_for_prediction1['s_inf_to_numpy']

    #Q: why is this not multiplied by the nr of waves? A: it is later in this function, this computation is all per wave initially.
    server_forward_pass = times_for_prediction1['inference_time2'][:idx_potential].sum()

    nr_waves = math.ceil(client_batch/512)
    par_last_wave = min(4, client_batch/128)  #assume client_batch multiple of 512 if > 512


    if output_size_layer < 4*1e8: #400.000.000 = 400MB
        s_inf_copy_to_cpu_bw = 2.38103186e-12 * nb_inferences
    else:
        s_inf_copy_to_cpu_bw = 7.52775484e-12 * nb_inferences
    #s_inf_copy_to_cpu = output_size_layer * s_inf_copy_to_cpu_bw /min(4, client_batch/128)
    s_inf_copy_to_cpu = output_size_layer * s_inf_copy_to_cpu_bw / nr_waves / par_last_wave
    
    #done in parallel on CPU on the SERVER up to 4 threads at a time. On the critical path. We count here 1 "wave" only. The rest are part of the wait_time.
    #s_pickle_time = output_size_layer * 9.00549092e-10 / min(4, client_batch/128)     
    s_pickle_time = output_size_layer * 9.00549092e-10 / nr_waves / par_last_wave


    #c_stream_pickle_load = output_size_layer * 4.76287825e-10 / nr_waves    
    #c_stream_dataloader = output_size_layer * 4.57389807e-10  

    #this bw is PER REQUEST so it already indirectly includes the impact of concurrency (so we only divide by bw). We count here for 1 "wave" only. 
    #s_sending_time = output_size_layer / (bw) / min(4, client_batch/128)  
    s_sending_time = output_size_layer / nr_waves / (bw * par_last_wave)           
         
    old_sending_time = output_size_layer/(12000.2855256674147 * 1024 * 1024 / 8)

    c_stream_from_server = s_read_data_shm+s_read_model_shm_to_gpu+s_inf_to_pytorch+s_inf_copy_to_gpu+s_inf_to_numpy+server_forward_pass+s_inf_copy_to_cpu+s_pickle_time+s_sending_time
    wait_time = (nr_waves - 1) * c_stream_from_server

    print("Estimated server times ", s_read_data_shm, s_read_model_shm_to_gpu, s_inf_to_pytorch, s_inf_copy_to_gpu, s_inf_to_numpy, server_forward_pass, s_inf_copy_to_cpu, s_pickle_time, 
        s_sending_time, c_stream_from_server)
    print("Estimated c_stream_from_server ", idx_potential, c_stream_from_server)

    c_stream_total_iteration = c_stream_from_server + wait_time
    #c_stream_total_iteration = c_stream_from_server + c_stream_pickle_load + c_stream_dataloader + wait_time

    print("Estimated c_stream_total_iteration ", c_stream_total_iteration)

    print("Estimated SRV times XX ", idx_potential, round(s_read_data_shm,5), round(s_read_model_shm_to_gpu,5), round(s_inf_to_pytorch,5), round(s_inf_copy_to_gpu,5), round(s_inf_to_numpy,5), round(server_forward_pass,5), round(s_inf_copy_to_cpu,5), round(s_pickle_time,5), round(s_sending_time,5), round(old_sending_time,5), round(wait_time, 5), round(c_stream_total_iteration,5))
    #print("Estimated SRV times XX ", idx_potential, round(s_read_data_shm,5), round(s_read_model_shm_to_gpu,5), round(s_inf_to_pytorch,5), round(s_inf_copy_to_gpu,5), round(s_inf_to_numpy,5), round(server_forward_pass,5), round(s_inf_copy_to_cpu,5), round(s_pickle_time,5), round(s_sending_time,5), round(old_sending_time,5), round(c_stream_pickle_load,5), round(c_stream_dataloader,5), round(wait_time, 5), round(c_stream_total_iteration,5))
    return c_stream_total_iteration 



def choose_split_idx(model_str, model, freeze_idx, client_batch, split_choice, split_idx_manual, nwbw, device, sizes, oom_splits, early_profiling_split, times_for_prediction1={}, times_for_prediction2={}):
    print("IN SPLITTING ALGO", flush=True)
    print("SPLIT IDX CHOICE, split idx manual, freeze_idx: ", split_choice, split_idx_manual, freeze_idx, flush=True)
    split_idx = freeze_idx

    if split_choice == 'manual':
        split_idx = split_idx_manual
    elif split_choice == 'to_freeze':
        split_idx = freeze_idx
    elif split_choice == 'to_max':
        split_idx = np.argmax(sizes[:freeze_idx]) + 1
    elif split_choice == 'to_min':
        split_idx = np.argmin(sizes[:freeze_idx]) + 1
    ret_est = 0        
    
    if split_choice == 'automatic' or split_choice == 'nsg':
        if len(times_for_prediction1) > 0:
            #nwbw=get_nwbw_by_tcpdump2(int(min(4, client_batch/128)))
            nwbw=get_nwbw_by_tcpdump()
            print("Average bandwidth from tcpdump: ", nwbw/1024/1024)
            nb_inferences = 8 * int(math.ceil(client_batch/512))

            c_stream_total_iterations = []
            c_train_totals = []
            not_parallels= []
    
            received_nw_Bps=0
            #received_nw_Bps = 1024 * 1024 * times_for_prediction['server_bw']/times_for_prediction['server_bw_count']
            #bw = received_nw_Bps * min(client_batch/128, 4)
            #print("Received fromsrv network bandwidth ZZ in MB/s per request:", received_nw_Bps/1024/1024, " counted over : ", times_for_prediction['server_bw_count'])
            #print("Profiled total network bandwidth in MB/s: ", bw/1024/1024)

            #print("This model ",model_str," has number of layers: ",len(model.all_layers))
            print("Estimated CLI times XX idx c_pickle c_stream_dl c_train_dl c_train_copy_to_gpu c_train_fwd_pass c_train_back_pass c_train_total")
            #print("Estimated CLI times XX idx c_train_dl c_train_copy_to_gpu c_train_fwd_pass c_train_back_pass c_train_total")
            print("Estimated SRV times XX idx read_data_shm model_shm_to_gpu inf_to_pytorch inf_copy_to_gpu inf_to_numpy srv_fwd_pass inf_copy_to_cpu s_pickle net old_net wait_time TOTAL")
            #print("Estimated SRV times XX idx read_data_shm model_shm_to_gpu inf_to_pytorch inf_copy_to_gpu inf_to_numpy srv_fwd_pass inf_copy_to_cpu s_pickle net old_net c_pickle c_stream_dl wait_time TOTAL")
            for idx_potential in range(1,len(times_for_prediction1['inference_time'])):
                print("ESTIMATION FOR LAYER :", idx_potential)
                output_size_layer = sizes[idx_potential-1] * 1.00000033
                print("Estimated output size layer :", output_size_layer)

                c_stream_total_iteration = _estimate_c_stream_total_iteration(times_for_prediction1, times_for_prediction2, nb_inferences, idx_potential, output_size_layer, received_nw_Bps, client_batch, nwbw)
                c_stream_total_iterations.append(c_stream_total_iteration)
                
                c_train_total, not_parallel = _estimate_c_train_total(times_for_prediction1, times_for_prediction2, nb_inferences, idx_potential, output_size_layer, client_batch, early_profiling_split)
                c_train_totals.append(c_train_total)
                not_parallels.append(not_parallel)

            #nr_iterations=math.ceil(32000/client_batch)    
            max_stream_train=[]
            #max_stream_train =  np.max(np.vstack((np.array(c_stream_total_iterations), np.array(c_train_totals))), axis=0)
            print(c_stream_total_iterations)
            print(len(c_stream_total_iterations))
            print(c_train_totals)
            print(len(c_train_totals))
            print("-----------------")
    

            if split_choice == "automatic":
                for i in range(0, len(c_stream_total_iterations)):
                    max_stream_train.append(c_stream_total_iterations[i] 
                        + (max(c_stream_total_iterations[i],c_train_totals[i]) + not_parallels[i])*(30720/client_batch - 1) 
                        + not_parallels[i] + c_train_totals[i])

            if split_choice == "nsg":
                for i in range(0, len(c_stream_total_iterations)):
                    max_stream_train.append((c_stream_total_iterations[i] + c_train_totals[i] + not_parallels[i])*(30720/client_batch))
    


            #print("Estimation array QQQ: ", max_stream_train)
            #print("Length of estimation array: ",len(max_stream_train))
            max_stream_train=max_stream_train[:freeze_idx]
            split_idx_ = np.argmin(max_stream_train) + 1
            print ("Initially chosen split index: ", split_idx_)


            tmp_numpyarr = np.array(max_stream_train)
            print("QQQ estimation array -- min at pos ", split_idx_ - 1, " estimated runtime: ", int(tmp_numpyarr[split_idx_-1]), " ",end="")
            #for i in range(len(tmp_numpyarr.astype(int))):
            #    print (str(i)+":"+str(tmp_numpyarr.astype(int)[i])+" ",end="")
            #print("")

            tmp_numpyarr_int = tmp_numpyarr.astype(int)
            tmp_idx = np.argsort(tmp_numpyarr_int)
            for i in range(len(tmp_idx)):
                print (str(tmp_idx[i])+":"+str(tmp_numpyarr_int[tmp_idx[i]])+" ",end="")
            print ("")

            tmp_idx = np.argsort(tmp_numpyarr)
            print ("QQQ TOP ", end="")
            for i in range(5):
                print (str(tmp_idx[i]) + ":" + '{0:.3f}'.format(tmp_numpyarr[tmp_idx[i]]) + " ",end="")
            print ("") 

        
            oom_splits[0]=128
            oom_splits[1]=128            
            for i in range(len(tmp_idx)):
                print("BLAAA ",(tmp_idx[i] + 1)," ",oom_splits)
                if ((tmp_idx[i] + 1) in oom_splits):
                    print(tmp_idx[i] + 1, " is in oom_splits. Skipping !")
                    split_idx_ = -1
                else:
                    split_idx_ = tmp_idx[i] + 1
                    break

            if split_idx_ == -1:
                print ("BAD ! All splits are expected to OOM. Resettting to freeze_idx")
                split_idx_ = freeze_idx 

            print ("QQQ After OOM check chose split index: ", split_idx_ , " with runtime: ", tmp_numpyarr[split_idx_ - 1] )

            print("Diff between TOP 2: ",tmp_numpyarr_int[tmp_idx[1]] - tmp_numpyarr_int[tmp_idx[0]])

            print(split_idx_)
            if split_idx_ <= freeze_idx:
                split_idx = split_idx_
            #split_idx=17
            print(split_idx)
            ret_est = tmp_numpyarr[split_idx - 1]
    return split_idx, (0,0), ret_est


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
                print('\t\tGPU ID: ', i, ' uuid,utilization.gpu,utilization.memory: ', _get_gpu_stats_other(i))

                res_i = _get_gpu_stats(i)[0]
                mem_free.append(res_i[2] - res_i[1])
                mem_used.append(res_i[1])
            print(f"\t\tMemory occupied: {mem_used} Time: {time.time() - start_t}", flush=True)
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


def send_request(request_dict, server_ip):
    # This function sends a request to the intermediate server through its socket and wait for the result
    # request_dict is the dict object that should be sent to the server

    HOST = server_ip  # The server's hostname or IP address		
    PORT = 65432  # The port used by the server

    # process request_dict
    options = request_dict['meta'].union(request_dict['header'])
    request_dict = {}
    mode = 'split'
    for opt in options:
        contents = opt.split(":")
        request_dict[contents[0]] = contents[1]

        if contents[0] == 'Split-Idx' and int(contents[1]) < 0:
            mode = 'vanilla'
    # Create a socket and send the required options
    #  print("Sending request.....")
    sys.stdout.flush()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        while True:
            try:
                s.connect((HOST, PORT))
                break
            except socket.error:
                print ("socket.connect refused. Retrying")
                time.sleep(1)
        s.sendall(json.dumps(request_dict).encode('utf-8'))
        #print ("[send_request] Finished sendall call at time: ", time.time(), flush=True)

        if mode == 'split':
            server_times_len = struct.unpack('>I', recvall(s,4))[0]
            server_times = recvall(s, server_times_len)
            server_times = pickle.loads(server_times)
        else:
            server_times = {}

        raw_msglen = recvall(s, 8)
        print ("[send_request] First small message received at: ", time.time(), flush=True)

        try:
            msglen = struct.unpack('>Q', raw_msglen)[0]
            data = recvall(s, msglen)
        except:
            # send I size 4
            msglen = struct.unpack('>I', raw_msglen[:4])[0]
            rest = raw_msglen[4:]
            data = recvall(s, msglen, rest)

        print ("[send_request] Heavy transfer received at: ", time.time(), flush=True)


        if mode == 'split':
            server_bw_len = struct.unpack('>I', recvall(s,4))[0]
            server_bw = recvall(s, server_bw_len)
            server_bw = pickle.loads(server_bw)
            server_times['server_bw'] = server_bw

        print(f"Length of received data: {len(data)} and send_request done at: ", time.time(), flush=True)
        return data, server_times


def stream_batch(dataset_name, stream_dataset_len, swift, datadir, parent_dir, labels, transform, batch_size, lstart, lend, model, server_ip, shm, 
                          mode='vanilla', split_idx=100, srv_bs=16, mem_cons=(0, 0), sequential=False, CACHED=True, TRANSFORMED=True, ALL_IN_COS=False, NO_ADAPT=False):
    COMP_FILE_SIZE = COMP_FILE_SIZE_DATASET[dataset_name]
    if dataset_name == "imagenet" and batch_size < 1000:
        COMP_FILE_SIZE_DATASET[dataset_name]=128
        COMP_FILE_SIZE = COMP_FILE_SIZE_DATASET[dataset_name]
    stream_time = time.time()
    server_times = {}
    print("The mode is: ", mode)
    data_transferred=0
    avg_srv_times=[]
    indexes=[]

    if shm != None:
        shm_link = shared_memory.SharedMemory(name=shm)

    if mode == 'split':
        parallel_posts = int((lend - lstart) / COMP_FILE_SIZE)  # number of posts request to run in parallel
        post_step = int((lend - lstart) / parallel_posts)  # if the batch is smaller, it will be handled on the server
        lend = stream_dataset_len[dataset_name] if lend > stream_dataset_len[dataset_name] else lend
        print("stream_batch -- Start {}, end {}, post_step {}".format(lstart, lend, post_step))
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
                #bs_server = SERVER_BATCH
                bs_server = srv_bs
            opts = {"meta": {"Ml-Task:inference",
                             "dataset:" + dataset_name, "model:{}".format(model),
                             f"Batch-Size:{bs_server}",  # {}".format(int(cur_step//5)),
                             f"Training-Batch-Size:{batch_size}",  # {}".format(int(cur_step//5)),
                             #f"Batch-Size:{cur_step}",  # {}".format(int(cur_step//5)),
                             "start:{}".format(s), "end:{}".format(cur_end),
                             #            "Batch-Size:{}".format(post_step),
                             #            "start:{}".format(lstart),"end:{}".format(lend),
                             "Split-Idx:{}".format(split_idx),
                             #"Fixed-Mem:{}".format(mem_cons[0]),
                             #"Scale-BSZ:{}".format(mem_cons[1]),
                             "COMP_FILE_SIZE:{}".format(COMP_FILE_SIZE)},
                    "header": {"Parent-Dir:{}".format(parent_dir)}}
            #          obj_name = "{}/ILSVRC2012_val_000".format(parent_dir)+((5-len(str(s+1)))*"0")+str(s+1)+".JPEG"
            if CACHED and TRANSFORMED:
                obj_name = f"{parent_dir}/vals{s}e{s + COMP_FILE_SIZE}.PTB.zip"
            else:
                obj_name = f"{parent_dir}/vals{s}e{s + COMP_FILE_SIZE}.zip"
            #      obj_name = "{}/ILSVRC2012_val_000".format(parent_dir)+((5-len(str(lstart+1)))*"0")+str(lstart+1)+".JPEG"
            post_objects.append(SwiftPostObject(obj_name, opts))  # Create multiple posts

        read_bytes = 0
        ctr = 0

        print("Before submit Pool at: ", time.time())
        with concurrent.futures.ThreadPoolExecutor() as executor:
            print("stream_batch submitting send_request to ThreadPoolExecutor")
            futures = [executor.submit(send_request, post_obj.options, server_ip) for post_obj in post_objects]     #should implement with as_completed
            print("After submit Pool at: ", time.time())

            ctr = 0
            for fut in futures:
                if (ctr == 0):
                    res, server_times = fut.result()    #blocking call
                    ctr = ctr + 1
                    avg_srv_times.append(server_times['s_inf_forward_pass'])    #this appends an array
                else:
                    res, tmp_srv = fut.result()
                    avg_srv_times.append(tmp_srv['s_inf_forward_pass'])         #this appends an array

                read_bytes += int(len(res))
                print("PID: {} Future #{} took {} sec transferred {} MB at time {}".format(os.getpid(), ctr + 1, time.time() - stream_time, int(len(res))/1024/1024, time.time()))

                if shm != None: #default case now
                    shm_link.buf[read_bytes - int(len(res)): read_bytes] = res
                else:
                    pickle_start_time = time.time()
                    images.extend(pickle.loads(res))
                    pickle_duration = time.time() - pickle_start_time
                    len_read_mb = int(len(res)/1024/1024)
                    print("Pickle took {} seconds for {} MB throughput {} at: {}".format(pickle_duration, len_read_mb, int(len_read_mb/pickle_duration), time.time()), flush=True)
                ctr = ctr + 1

        avg_srv_times=np.mean(avg_srv_times[0:], axis=0)    #mean across all those arrays, one array per request to server (i.e. future call)

        transform = None  # no transform required in this case
        print("Read {} MB for this batch".format(read_bytes / (1024 * 1024)), flush=True)
        print("Executing all futures took {} seconds at: {}".format(time.time() - post_time, time.time()), flush=True)

    else:  # mode=='vanilla'
        if not CACHED:
            print ("Path not taken")
        elif CACHED:
            parallel_posts = int((lend - lstart) / COMP_FILE_SIZE)  # number of posts request to run in parallel
            post_step = int((lend - lstart) / parallel_posts)  # if the batch is smaller, it will be handled on the server
            lend = stream_dataset_len[dataset_name] if lend > stream_dataset_len[dataset_name] else lend
            print("Start {}, end {}, post_step {}".format(lstart, lend, post_step))
            post_objects = []
            images = []
            post_time = time.time()
            for i, s in enumerate(range(lstart, lend, post_step)):
                cur_end = s + post_step if s + post_step <= lend else lend
                cur_step = cur_end - s
                opts = {"meta": {"Ml-Task:inference",
                                 "dataset:" + dataset_name, "model:{}".format(model),
                                 f"Batch-Size:{COMP_FILE_SIZE}",  # {}".format(int(cur_step//5)),
                                 f"Training-Batch-Size:{batch_size}",  # {}".format(int(cur_step//5)),
                                 "start:{}".format(s), "end:{}".format(cur_end),
                                 #            "Batch-Size:{}".format(post_step),
                                 #            "start:{}".format(lstart),"end:{}".format(lend),
                                 "Split-Idx:{}".format(-1),
                                 #"Fixed-Mem:{}".format(mem_cons[0]),
                                 #"Scale-BSZ:{}".format(mem_cons[1]),
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
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                print("stream_batch submitting send_request to ThreadPoolExecutor for BASELINE")
                futures = [executor.submit(send_request, post_obj.options, server_ip) for post_obj in post_objects]     #should implement with as_completed
                print("After submit Pool at: ", time.time())

                ctr = 0
                #img=[]
                for fut in futures:
                    res, _ = fut.result()
                    read_bytes += int(len(res))
                    print("PID: {} Future #{} took {} sec transferred {} MB at time {}".format(os.getpid(), ctr + 1, time.time() - stream_time, int(len(res))/1024/1024, time.time()))

                    if shm != None: #default case now
                        indexes.append(int(len(res)))
                        shm_link.buf[read_bytes - int(len(res)): read_bytes] = res    
                        #f2 = BytesIO(res)
                        #zipff = zipfile.ZipFile(f2, 'r')
                        #print ("YYYY ",len(res))
                        #print (zipff.infolist())
                        #print (zipff)
                        
                        #img.extend(np.array(torch.load(io.BytesIO(zipff.open(f3).read()))) for f3 in zipff.infolist())
                        #print(img)
                    ctr = ctr + 1
            print("Read {} MB for this batch".format(read_bytes / (1024 * 1024)), flush=True)
            print("Executing all futures took {} seconds at: {}".format(time.time() - post_time, time.time()), flush=True)        
                        
    if (shm != None):   #default case now
        dataloader = None
        shm_link.close()
    else:
        # use only labels corresponding to the required images
        labels = labels[lstart:lend]
        assert len(images) == len(labels)
    
        dataloader_start_time = time.time()
        imgs = np.array(images)
        dataset = InMemoryDataset(imgs, labels=labels, transform=transform, mode=mode, transformed=TRANSFORMED)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        dataloader_duration = time.time() - dataloader_start_time
        len_read_mb = int(read_bytes/1024/1024)

        print("Dataloader took {} seconds for {} MB throughput {} at: {}".format(dataloader_duration, len_read_mb, int(len_read_mb/dataloader_duration),time.time()), flush=True)

    server_times['avg_srv_times']=avg_srv_times
    print("Streaming {} data took {} seconds".format(dataset_name, time.time() - stream_time))

    return dataloader, server_times, read_bytes, indexes





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

