#These are some important functions that are copied from PyTorch implementation of MNIST processing
# https://pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html#MNIST
import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
import string
import gzip
import lzma
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union

def get_int(b: bytes) -> int:
    return int(codecs.encode(b, 'hex'), 16)


def open_maybe_compressed_file(path: Union[str, IO]) -> Union[IO, gzip.GzipFile]:
    """Return a file object that possibly decompresses 'path' on the fly.
       Decompression occurs when argument `path` is a string and ends with '.gz' or '.xz'.
    """
    if not isinstance(path, torch._six.string_classes):
        return path
    if path.endswith('.gz'):
        return gzip.open(path, 'rb')
    if path.endswith('.xz'):
        return lzma.open(path, 'rb')
    return open(path, 'rb')


SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype('>i2'), 'i2'),
    12: (torch.int32, np.dtype('>i4'), 'i4'),
    13: (torch.float32, np.dtype('>f4'), 'f4'),
    14: (torch.float64, np.dtype('>f8'), 'f8')
}


def read_sn3_pascalvincent_tensor(path: Union[str, IO], strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open_maybe_compressed_file(path) as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


def read_label_file(path: str) -> torch.Tensor:
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    with open(path, 'rb') as f:
        x = read_sn3_pascalvincent_tensor(f, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x

def process_mnist(datadir):
  ###########################################################################################################
  #This is the work that should be done to prepare the MNIST dataset...
  #the same piece of code is excted in native PyTorch code
  training_set = (
        read_image_file(os.path.join(datadir,'mnist', 'train-images-idx3-ubyte')),
        read_label_file(os.path.join(datadir,'mnist', 'train-labels-idx1-ubyte'))
  )
  test_set = (
        read_image_file(os.path.join(datadir, 'mnist', 't10k-images-idx3-ubyte')),
        read_label_file(os.path.join(datadir, 'mnist', 't10k-labels-idx1-ubyte'))
  )
  processed_folder = os.path.join('MNIST','processed')	#this format is required by PyTorch code
  training_file = 'training.pt'
  test_file = 'test.pt'
  os.makedirs(processed_folder, exist_ok=True)
  with open(os.path.join(datadir, processed_folder, training_file), 'wb') as f:
    torch.save(training_set, f)
  with open(os.path.join(datadir, processed_folder, test_file), 'wb') as f:
    torch.save(test_set, f)
###########################################################################################################
