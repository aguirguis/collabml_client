#This script takes multiple Imagenet images and put them in one zip file
#Currently, we compress each 500 images together
import os
from os.path import expanduser
import io
import numpy as np
from PIL import Image
import codecs
from io import BytesIO
home=expanduser("~")
#prepare files
for i in range(0,50000):
    idtostr = lambda idx: "val/ILSVRC2012_val_000"+((5-len(str(idx+1)))*"0")+str(idx+1)+".JPEG"
    file_image = os.path.join(home, "dataset/imagenet", idtostr(i))
    convert = lambda file: np.array(Image.open(io.BytesIO(file.read())).convert('RGB'))
    with open(file_image, "rb") as fi:
            image = convert(fi)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    image = transform(image)

    image_buffer = io.BytesIO()
    torch.save(x, image_buffer)

    with open(file_image.replace(".JPEG",".PTB"), "wb") as f:
        f.write(image_buffer)
    break
