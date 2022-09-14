#!/usr/bin/env python3
#This library include independent functions to parse the output files
def get_total_exec_time(filenames):
    #param filenames: list of output file names to be parsed
    #returns times: list of total execution time (len(times)=len(filenames)) of each output file
    times = []
    for filename in filenames:
        print(f"Processing file: {filename}")
        with open(filename, "r", encoding='latin1') as f:
            lines = f.readlines()
            if len(lines) == 0:
                times.append(0)
                continue
            line = lines[0]
            if not line.startswith("The whole"):
                for l in lines:
                    if l.startswith("RuntimeError: CUDA out of memory.") or l.startswith("Exception:"):
                        line = l
                        break
                    if l.startswith("The whole"):
                        line = l
                        break
            try:
                assert line.startswith("The whole")
                times.append(float(line.split()[-2]))
            except Exception as e:	#This has probably crashed with OOM
                times.append(0)
    return times

def get_split_idx(filenames):
    #param filenames: list of output file names to be parsed
    #returns idxs: list of split idxs chosen by different output files (len(idxs)=len(filenames))
    idxs = []
    for filename in filenames:
        print(f"Processing file: {filename}")
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Using split index:"):
                    idxs.append(int(line.split()[-1]))
                    break
    return idxs

def get_output_size(filenames):
    #param filenames: list of output file names to be parsed
    #returns outputs: list of 'intermediate' output sizes in different output files (len(outputs)=len(filenames))
    #Note: the result is returned in MBs
    outputs = []
    for filename in filenames:
        print(f"Processing file: {filename}")
        with open(filename, "r") as f:
            lines = f.readlines()
            sizes = []
            for line in lines:
                if line.startswith("Read"):
                    sizes.append(float(line.split()[1]))
            outputs.append(max(sizes))
    return outputs

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

def get_batch_size_dec(filenames):
    #param filenames: list of output file names to be parsed
    #returns bs_dec: list of tuples: (requested_batch_size, decided_batch_size) in each output file (len(bs_dec)=len(filenames))
    #Note that: this is used only with the server logs (rather than the client)
    bs_dec = []
    for filename in filenames:
        print(f"Processing file: {filename}")
        with open(filename, "r") as f:
            lines = f.readlines()
            bs = []
            for line in lines:
                if line.startswith("The requested batch"):
                    a = line.split()
                    bs.append((int(a[4][:-1]),int(a[-1])))
            bs_dec.append(bs)
    return bs_dec

def get_percent_mismatch_bs(filenames):
    #param filenames: list of output file names to be parsed
    #returns bs_mismatch: list of percentage of mismatch between the requested and the decided batch size in each output file (len(bs_mismatch)=len(filenames))
    #Note that: this is used only with the server logs (rather than the client)
    bs_decs = get_batch_size_dec(filenames)
    bs_mismatch = []
    for bc_dec in bs_decs:
        mismatch = 0
        for dec in bc_dec:
            if dec[0] != dec[1]:
                mismatch+=1
        bs_mismatch.append(mismatch*100.0/len(bc_dec) if len(bc_dec) != 0 else 0)
    return bs_mismatch

def get_reduction_bs(filenames):
    #param filenames: list of output file names to be parsed
    #returns bs_red: list of average reduction in the batch size (if any) that is decided by the batch adaptation algorithm in each output file (len(bs_red)=len(filenames))
    #Note that: this is used only with the server logs (rather than the client)
    bs_decs = get_batch_size_dec(filenames)
    bs_red = []
    for bc_dec in bs_decs:
        reds=[]
        for dec in bc_dec:
            reds.append((dec[0]-dec[1])*100.0/dec[0])	#reduction = (initial_val - final_val)/initial_val
        bs_red.append(sum(reds)/len(reds) if len(reds) != 0 else 0)		#average of reductions
    return bs_red
