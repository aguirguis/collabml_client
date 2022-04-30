import os
#import matplotlib.pyplot as plt
import numpy as np
import PIL
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch import Tensor

from timm import create_model
from timm.models import vision_transformer
from timm.models.layers import PatchEmbed
from timm.models.helpers import checkpoint_seq

import time

from PIL import Image
from torch.utils.data import Dataset

from functools import partial

def flatten(model: torch.nn.Module):
    children = list(model.children())
    flatt_children = []
    if children == []:
        return model
    else:
        for child in children:
            if isinstance(child, vision_transformer.Block):
                # consider a block as a single layer
                flatt_children.append(child)
            else:
                try:
                    flatt_children.extend(flatten(child))
                except TypeError:
                    flatt_children.append(flatten(child))
    return flatt_children

class MyViT(vision_transformer.VisionTransformer):
    features_size_block = {}

    def forward(self, x, start=0, end=10000, need_time=False):
        res = []
        time_res = []
        layer_time = time.time()

        nbLayer = -1
        blocks_done = False

        for child in self.children():
            if isinstance(child, PatchEmbed):
                updated = False
                lcc = flatten(child)
                for cc in lcc:
                    nbLayer += 1
                    if start <= nbLayer < end:
                        layer_time = time.time()
                        x = cc(x)
                        updated = True
                        if isinstance(cc, torch.nn.Conv2d):
                            updated = False
                            if child.flatten:
                                x = x.flatten(2).transpose(1, 2)
                            time_res.append(time.time() - layer_time)
                            if isinstance(cc, torch.nn.Dropout) or isinstance(cc, torch.nn.Identity):
                                res.append(0)
                            else:
                                res.append(x.element_size() * x.nelement() / (1024 ** 2))

                if updated:
                    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                    x = x + self.pos_embed
                    time_res.append(time.time() - layer_time)
                    if isinstance(cc, torch.nn.Dropout) or isinstance(cc, torch.nn.Identity):
                        res.append(0)
                    else:
                        res.append(x.element_size() * x.nelement() / (1024 ** 2))
            else:
                lcc = flatten(child)
                if isinstance(lcc, torch.nn.Module):
                    nbLayer += 1
                    if start <= nbLayer < end:
                        layer_time = time.time()
                        # single layer
                        x = child(x)
                        if isinstance(child, torch.nn.modules.LayerNorm) and blocks_done:
                            if self.global_pool:
                                x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
                        time_res.append(time.time() - layer_time)
                        if isinstance(cc, torch.nn.Dropout) or isinstance(cc, torch.nn.Identity):
                            res.append(0)
                        else:
                            res.append(x.element_size() * x.nelement() / (1024 ** 2))
                else:
                    for cc in lcc:
                        nbLayer += 1
                        if start <= nbLayer < end:
                            layer_time = time.time()
                            x = cc(x)
                            time_res.append(time.time() - layer_time)
                            if isinstance(cc, vision_transformer.Block):
                                # TODO change afterwards
                                sum_sizes = sum(self.features_size_block[list(self.features_size_block.keys())[0]])
                                #res.append(16594.171875)
                                #print(sum_sizes)
                                res.append(sum_sizes)
                            else:
                                if isinstance(cc, torch.nn.Dropout) or isinstance(cc, torch.nn.Identity):
                                    res.append(0)
                                else:
                                    res.append(x.element_size() * x.nelement() / (1024 ** 2))
                    if isinstance(child, torch.nn.Sequential):
                        blocks_done = True

        if need_time:
            return x, torch.Tensor(res).cuda(), time_res
        return x, torch.Tensor(res).cuda()


def build_my_vit(num_classes=10):
    def get_features(i, features_size):
        def hook(model, input, output):
            if i in features_size:
                # if i == 0:
                #     print(model)
                #     print("torch cuda memory allocated ", torch.cuda.memory_allocated(0) / (1024 ** 2))
                #     print("estimated += ", output.element_size() * output.nelement() / (1024 ** 2))
                features_size[i].append(output.element_size() * output.nelement() / (1024 ** 2))
            else:
                features_size[i] = [output.element_size() * output.nelement() / (1024 ** 2)]
        return hook

    m = MyViT(num_classes=num_classes)
    for i, block in enumerate(m.blocks):
        for lay in flatten(block):
            if not (isinstance(lay, torch.nn.Dropout) or isinstance(lay, torch.nn.Identity)):
                lay.register_forward_hook(get_features(i, m.features_size_block))
    return m


import subprocess
import shutil
import threading
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

#from gpu_mem_track import MemTracker
#gpu_t = MemTracker()
def print_stats(m):
    # gpu_t.track()
    print(m)
    print("nvidia-smi gpu utilization ", _get_gpu_stats(0)[0][1])
    print("torch cuda memory allocated ", torch.cuda.memory_allocated(0) / (1024 ** 2))
    print("torch cuda max memory allocated ", torch.cuda.max_memory_allocated(0) / (1024 ** 2))
    print("torch cuda memory reserved ", torch.cuda.memory_reserved(0) / (1024 ** 2))
    #print(torch.cuda.max_memory_reserved(0) / (1024 ** 2))
    print()

batch_sizes = [100]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Define transforms for test
# IMG_SIZE = (224, 224)
# NORMALIZE_MEAN = (0.5, 0.5, 0.5)
# NORMALIZE_STD = (0.5, 0.5, 0.5)
# transforms = [
#     T.Resize(IMG_SIZE),
#     T.ToTensor(),
#     T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
# ]
# transforms = T.Compose(transforms)

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transforms = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
])

img = PIL.Image.open('santorini.png')

for batch_size in batch_sizes:
    torch.cuda.empty_cache()
    print_stats("Beginning of exp")
    img_tensor = transforms(img).unsqueeze(0).to(device)
    img_tensor = img_tensor.repeat(batch_size, 1, 1, 1)
    input_size = img_tensor.element_size() * img_tensor.nelement() / (1024**2)
    print("Input size ", input_size)

    bs = img_tensor.size()[0]
    print_stats(f"After loading input to cuda, input batch size {bs}")

    model_test = build_my_vit(1000)
    model_time = time.time()
    if torch.cuda.is_available():
        model_test.cuda()
    print("Time to load model: {}\r\n".format(time.time()-model_time))
    params=[param for param in model_test.parameters()]
    mod_sizes = [np.prod(np.array(p.size())) for p in params]
    model_size = np.sum(mod_sizes)*4/ (1024*1024)
    print("Model_size: ", model_size)
    print_stats("After loading model to cuda ")

    inference_time = time.time()
    model_test.eval()
    output, res = model_test(img_tensor)
    print("Time for inference: {}\r\n".format(time.time() - model_time))
    print("Layer sizes, total sum", res, sum(res))
    print("Expected (input+layers+model): ", input_size+sum(res)+model_size)
    print_stats("After inference eval mode")


    model_test.train()
    output, res = model_test(img_tensor)
    print("Layer sizes, total sum", res, sum(res))
    print("Expected (input+2*layers+model): ", input_size+2*sum(res)+model_size)
    print_stats("After inference train mode")

    print()