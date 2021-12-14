import os

def get_gpu_mem_cons_time(filenames):
    #param filenames: list of output file names to be parsed
    #returns gpu_mems: list of lists: the total gpu consumption (on both GPUs) with time (len(gpu_mems)=len(filenames))
    #Note that this can be used in either client or server sides
    gpu_mems = []
    for filename in filenames:
        print(f"Processing file: {filename}")
        with open(filename, "r") as f:
            lines = f.readlines()
            mem_inst = []
            for line in lines:
                if line.startswith("Memory occpied:"):
                    a = line.split()
                    mem_inst.append(float(a[2][1:-1]) + float(a[3][:-1]))
            gpu_mems.append(mem_inst)
    return gpu_mems

def get_gpu_mem_cons(filenames):
    #param filenames: list of output file names to be parsed
    #returns gpu_mems: list of highest memory consumption (over time) in each output file (len(gpu_mems)=len(filenames))
    #Note that this can be used in either client or server sides
    gpu_mems = get_gpu_mem_cons_time(filenames)
    gpu_mems = [max(mem, default=0) for mem in gpu_mems]
    return gpu_mems

######Main script
filenames = []
for split in range(11,29):
    freeze = 25 if split < 25 else split
    os.system(f"python3 /root/swift_playground/application_layer/main.py --dataset imagenet\
		 --model myvgg11 --num_epochs 1 --batch_size 1000 --freeze --freeze_idx {freeze}\
		 --use_intermediate --split_idx {split} --manual_split | tee temp/vgg11_manual_split_{split}")
    filenames.append(f"temp/vgg11_manual_split_{split}")
os.system("python3 /root/swift_playground/application_layer/main.py --dataset imagenet\
	 --model vgg11 --num_epochs 1 --batch_size 1000 --freeze --freeze_idx 25 | tee temp/vanilla_vgg11")
filenames.append("temp/vanilla_vgg11")

print(get_gpu_mem_cons(filenames))
