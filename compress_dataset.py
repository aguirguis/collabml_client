#This script takes multiple Imagenet images and put them in one zip file
#Currently, we compress each 500 images together
import zipfile
import os
from os.path import expanduser
import io
home=expanduser("~")

dataset = 'imagenet'

dataset_steps = {
        'imagenet': [128, 1000],
        'plantleave': [50],
        'inaturalist': [250]
}

dataset_total_images = {
        'imagenet': 24320, #50000,
        'plantleave': 4502,
        'inaturalist': 24426
}

#prepare files
steps = dataset_steps[dataset]
total_images = dataset_total_images[dataset]
extensions = [".JPEG", ".PTB"]

dataset_path = os.path.join(home, 'dataset/', dataset)
if not os.path.isdir(dataset_path+'/compressed'):
    os.system(f"mkdir {dataset_path}/compressed")
os.system(f"cp {dataset_path}/val/ILSVRC2012_validation_ground_truth.txt {dataset_path}/compressed/ILSVRC2012_validation_ground_truth.txt")
for ext in extensions:
    for step in steps:
        for i in range(0,total_images,step):
            start, end = i,i+step if i+step < total_images else total_images
            print(f"Compressing images from {start} to {end}")
            idtostr = lambda idx: "val/ILSVRC2012_val_000"+((5-len(str(idx+1)))*"0")+str(idx+1)+ext
            objects = [idtostr(idx) for idx in range(start,end)]
            list_files = [os.path.join(home, "dataset/"+dataset,obj) for obj in objects]
            #   compress
            path_compressed = f'vals{start}e{end}'+ext+'.zip' if ext != ".JPEG" else f'vals{start}e{end}.zip'
            with zipfile.ZipFile(os.path.join(dataset_path, 'compressed', path_compressed), 'w') as zipf:
                for file in list_files:
                    zipf.write(file, compress_type=zipfile.ZIP_DEFLATED)

