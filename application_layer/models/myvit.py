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
    features_size_block = []

    def forward(self, x, start=0, end=10000, need_time=False):
        if need_time == False:
            return super().forward(x)
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
                    layer_time = time.time()
                    if start <= nbLayer < end:
                        x = self.forward_layer(cc, layer_time, res, time_res, x)
                        updated = True
                        if isinstance(cc, torch.nn.Conv2d):
                            updated = False
                            if child.flatten:
                                x = x.flatten(2).transpose(1, 2)

                if updated:
                    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                    x = x + self.pos_embed
            else:
                lcc = flatten(child)
                if isinstance(lcc, torch.nn.Module):
                    nbLayer += 1
                    layer_time = time.time()
                    if start <= nbLayer < end:
                        # single layer
                        x = self.forward_layer(child, layer_time, res, time_res, x)
                        if isinstance(child, torch.nn.modules.LayerNorm) and blocks_done:
                            if self.global_pool:
                                x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
                else:
                    for cc in lcc:
                        nbLayer += 1
                        layer_time = time.time()
                        if start <= nbLayer < end:
                            if isinstance(cc, vision_transformer.Block):
                                x = self.forward_layer_block(cc, layer_time, res, time_res, x)
                            else:
                                x = self.forward_layer(cc, layer_time, res, time_res, x)
                    if isinstance(child, torch.nn.Sequential):
                        blocks_done = True

        if need_time:
            return x, torch.Tensor(res), time_res
        return x, torch.Tensor(res)

    def forward_layer_block(self, cc, layer_time, res, time_res, x):
        x = cc(x)
        time_res.append(time.time() - layer_time)
        res.append(sum(self.features_size_block))
        return x

    def forward_layer(self, cc, layer_time, res, time_res, x):
        x = cc(x)
        time_res.append(time.time() - layer_time)
        res.append(x.element_size() * x.nelement() / 1024)
        return x


def build_my_vit(num_classes=10):
    def get_features(features_size):
        def hook(model, input, output):
            features_size.append(output.element_size() * output.nelement() / 1024)
        return hook
    m = MyViT(num_classes=num_classes)
    for lay in flatten(m.blocks[0]):
        lay.register_forward_hook(get_features(m.features_size_block))
    return m
