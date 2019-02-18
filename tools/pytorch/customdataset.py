import functools
import itertools as it
import operator

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, Subset


class DownsampleDataset(Subset):
    def __init__(self, dataset, proportion=0.01, n_samples=None):
        n_samples_dataset = len(dataset)
        if n_samples is None:
            n_samples = int(n_samples_dataset * proportion)
        indices = np.random.randint(n_samples_dataset, size=(n_samples,)).tolist()
        super(DownsampleDataset, self).__init__(dataset, indices)


class RemoveDatasetColumns(Dataset):
    def __init__(self, dataset, index=1):
        self.dataset = dataset
        self.index = [index] if isinstance(index, int) else index
        self.index = sorted(self.index, reverse=True)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        out = list(self.dataset[idx])
        for i in self.index:
            del(out[i])
        return out[0] if len(out) == 1 else out
        

class SelectDatasetColumns(Dataset):
    def __init__(self, dataset, index=0):
        self.index = [index] if isinstance(index, int) else index
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        out = self.dataset[idx]
        return operator.itemgetter(*self.index)(out)


class SyncedDatasets(Dataset):
    def __init__(self, *dataset):
        assert(len(set(map(len, dataset))) == 1), "All datasets should be of same length"
        self.datasets = dataset  # datasets should be aligned!

    def __getitem__(self, index):
        items = map(operator.itemgetter(index), self.datasets)
        return tuple(it.chain(*(i if isinstance(i, tuple) else (i,)
                                for i in items)))

    def __len__(self):
        return len(self.datasets[0])


def make_dataloader(dataset, batch_size=2, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=3, pin_memory=torch.cuda.is_available())


class AddConstLabelToDataset(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx], self.label


class LambdaDataset(Dataset):
    def __init__(self, dataset,*func, at_index=None):
        self.dataset = dataset
        self.func = func
        self.at_index = at_index
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        #return self.func(self.dataset[idx])
        if not self.at_index:
            return functools.reduce(lambda x, f: f(x), self.func, self.dataset[idx])
        # print("labmda:", self.func)
        temp_ret = list(self.dataset[idx])
        temp_ret[self.at_index] = functools.reduce(lambda x, f: f(x), self.func, temp_ret[self.at_index])
        return tuple(temp_ret)


if __name__ == "__main__":
    # np.random.seed(1212)
    from torchvision.datasets import MNIST
    a = MNIST(root='.', train=True, download=True)
    b = DownsampleDataset(a, proportion=0.1, n_samples=500)
    # c = RemoveDatasetColumns(b, index=1)
    d = SelectDatasetColumns(b, index=[0])

    # #d = make_dataloader(c, batch_size=2, shuffle=False)
    # print(next(iter(d)))
    e = MakeLabeledDataset(d, 1)
    print(next(iter(e)))