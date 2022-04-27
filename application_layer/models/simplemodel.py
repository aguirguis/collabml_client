import os
import numpy as np

import torch
import torch.nn as nn

import subprocess
import shutil

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(10, 20, 5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(20, 64, 5)
        self.relu2 = nn.ReLU()
    def forward(self, x):
        res = []
        #res.append(x.element_size() * x.nelement() / (1024**2))
        for layer in self.children():
            x = layer(x)
            res.append(x.element_size() * x.nelement() / (1024**2))
        return x, torch.Tensor(res).cuda()


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

batch_sizes = [1, 10, 100]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for batch_size in batch_sizes:
    torch.cuda.empty_cache()
    print_stats("Beginning of exp")
    img_tensor = torch.rand((batch_size, 10, 30, 20)).to(device)
    input_size = img_tensor.element_size() * img_tensor.nelement() / (1024**2)
    print("Input size ", input_size)

    bs = img_tensor.size()[0]
    print_stats(f"After loading input to cuda, input batch size {bs}")

    model_test = MyModel()
    if torch.cuda.is_available():
        model_test.cuda()

    params=[param for param in model_test.parameters()]
    mod_sizes = [np.prod(np.array(p.size())) for p in params]
    model_size = np.sum(mod_sizes)*4/ (1024*1024)
    print("Model_size: ", model_size)
    print_stats("After loading model to cuda ")

    model_test.eval()
    output, res = model_test(img_tensor)
    print("Layer sizes, total sum", res, sum(res))
    print("Expected (input+layers+model): ", input_size+sum(res)+model_size)
    print_stats("After inference eval mode")


    model_test.train()
    output, res = model_test(img_tensor)
    print("Layer sizes, total sum", res, sum(res))
    print("Expected (input+2*layers+model): ", input_size+2*sum(res)+model_size)
    print_stats("After inference train mode")

    print()
