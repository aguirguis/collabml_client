#This script takes multiple Imagenet images and put them in one zip file
#Currently, we compress each 500 images together
import zipfile
import os
from os.path import expanduser
import io
home=expanduser("~")
#prepare files
step = 500
for i in range(0,4502,step):
    start, end = i,i+step if i+step < 4502 else 4502
    print(f"Compressing images from {start} to {end}")
    #idtostr = lambda idx: "val/ILSVRC2012_val_000"+((5-len(str(idx+1)))*"0")+str(idx+1)+".PTB"
    idtostr = lambda idx: "val/ILSVRC2012_val_000"+((5-len(str(idx+1)))*"0")+str(idx+1)+".JPEG"
    objects = [idtostr(idx) for idx in range(start,end)]
    list_files = [os.path.join(home, "dataset/plantleave",obj) for obj in objects]
    #   compress
    with zipfile.ZipFile(os.path.join(home, "dataset/plantleave/compressed",f'vals{start}e{end}.zip'), 'w') as zipf:
        for file in list_files:
            zipf.write(file, compress_type=zipfile.ZIP_DEFLATED)
