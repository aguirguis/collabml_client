import torch
from torch.utils.data import Dataset

#Wrapper to datasets already loaded in memory
class InMemoryDataset(Dataset):
    """In memory dataset wrapper."""

    def __init__(self, dataset, transform=None):
        """
        Args:
            dataset (ndarray): array of dataset samples
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.dataset[idx]
        if self.transform:
            image = self.transform(image)

        return image
