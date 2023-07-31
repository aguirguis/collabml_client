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
        memory_res = []
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
                            torch.cuda.synchronize()
                            time_res.append(time.time() - layer_time)
                            res.append(x.element_size() * x.nelement() / 1024)
                            if isinstance(cc, torch.nn.Dropout) or isinstance(cc, torch.nn.Identity):
                                memory_res.append(0)
                            else:
                                memory_res.append(x.element_size() * x.nelement() / 1024)

                if updated:
                    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                    x = x + self.pos_embed
                    torch.cuda.synchronize()
                    time_res.append(time.time() - layer_time)
                    res.append(x.element_size() * x.nelement() / 1024)
                    if isinstance(cc, torch.nn.Dropout) or isinstance(cc, torch.nn.Identity):
                        memory_res.append(0)
                    else:
                        memory_res.append(x.element_size() * x.nelement() / 1024)
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
                        torch.cuda.synchronize()
                        time_res.append(time.time() - layer_time)
                        res.append(x.element_size() * x.nelement() / 1024)
                        if isinstance(cc, torch.nn.Dropout) or isinstance(cc, torch.nn.Identity):
                            memory_res.append(0)
                        else:
                            memory_res.append(x.element_size() * x.nelement() / 1024)
                else:
                    for cc in lcc:
                        nbLayer += 1
                        if start <= nbLayer < end:
                            layer_time = time.time()
                            x = cc(x)
                            torch.cuda.synchronize()
                            time_res.append(time.time() - layer_time)
                            res.append(x.element_size() * x.nelement() / 1024)
                            if isinstance(cc, vision_transformer.Block):
                                sum_sizes = sum(self.features_size_block[list(self.features_size_block.keys())[0]])
                                memory_res.append(sum_sizes)
                            else:
                                if isinstance(cc, torch.nn.Dropout) or isinstance(cc, torch.nn.Identity):
                                    memory_res.append(0)
                                else:
                                    memory_res.append(x.element_size() * x.nelement() / 1024)
                    if isinstance(child, torch.nn.Sequential):
                        blocks_done = True

        if need_time:
            return x, torch.Tensor(res).cuda(), time_res, memory_res, [idx for idx in range(start, start+len(res))]
        return x, torch.Tensor(res).cuda()


def build_my_vit(num_classes=10):
    def get_features(i, features_size):
        def hook(model, input, output):
            if i in features_size:
                features_size[i].append(output.element_size() * output.nelement() / 1024)
            else:
                features_size[i] = [output.element_size() * output.nelement() / 1024]
        return hook

    m = MyViT(num_classes=num_classes)
    for i, block in enumerate(m.blocks):
        for lay in flatten(block):
            if not (isinstance(lay, torch.nn.Dropout) or isinstance(lay, torch.nn.Identity)):
                lay.register_forward_hook(get_features(i, m.features_size_block))
    return m


#print(build_my_vit())
