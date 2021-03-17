import torch
from torch.utils.data import Dataset
import numpy
from PIL import Image

#Wrapper to datasets already loaded in memory
class InMemoryDataset(Dataset):
    """In memory dataset wrapper."""

    def __init__(self, dataset, labels=None, transform=None, mode='vanilla', logFile=None):
        """
        Args:
            dataset (ndarray): array of dataset samples
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.dataset = dataset
        self.labels = labels
        self.transform = transform
        self.logFile = logFile
        self.mode = mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.dataset[idx]
        if self.mode == 'split':		#this is not an image then, yet it is some intermediate result
            image = torch.from_numpy(image)
        else:
            try:
                image = Image.fromarray(image)
            except:
                image = Image.fromarray(image.numpy(), mode='L')
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            return image, int(self.labels[idx])
        return image

