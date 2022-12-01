#This script takes multiple Imagenet images and put them in one zip file
#Currently, we compress each 500 images together
import os
from os.path import expanduser
import io
import numpy as np
from PIL import Image
import codecs
from io import BytesIO
import torchvision.transforms as transforms
import torch

home=expanduser("~")

dataset = 'imagenet'

dataset_len = {'imagenet': 24320, #50000,
        'plantleave': 4502,
        'inaturalist': 24426
        }

#prepare files
for i in range(0, dataset_len[dataset]):
    idtostr = lambda idx: "val/ILSVRC2012_val_000"+((5-len(str(idx+1)))*"0")+str(idx+1)+".JPEG"
    file_image = os.path.join(home, "dataset/"+dataset, idtostr(i))
    convert = lambda file: Image.open(io.BytesIO(file.read())).convert('RGB')
    with open(file_image, "rb") as fi:
            image = convert(fi)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)

    torch.save(image, file_image.replace(".JPEG",".PTB"))

    #image_buffer = io.BytesIO()
    #torch.save(image, image_buffer)

    #with open(file_image.replace(".JPEG",".PTB"), "wb") as f:
    #    f.write(image_buffer.read())
