# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import Dataset


class Market1501(Dataset):
    def __init__(self, fpath, train=True, test=False, transform=None):
        with np.load(fpath) as data:
            if train:
                self.X = data['X_train']   # (N,H,W,C)
                self.y = data['y_train']
                self.pids = data['trainval_pids']
            elif test:
                self.X = data['X_query']
                self.y = data['y_query']
                self.cam_ids = data['cam_ids']
                self.X_distractors = data['X_distractors']
                self.y_distractors = data['y_distractors']
            else:
                self.X = data['X_val']
                self.y = data['y_val']
        
        assert len(self.X) == len(self.y)
        self.train = train
        self.test = test
        
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        inp, target = self.X[idx], self.y[idx]
        if self.transform is not None:
            inp = self.transform(inp)
        return inp, target



class CUHK03(Dataset):
    def __init__(self, fpath, train=True, test=False, transform=None):
        with np.load(fpath) as data:
            if train:
                self.X = data['X_train']   # (N,H,W,C)
                self.y = data['y_train']
                self.pids = data['trainval_pids']
            elif test:
                self.X = data['X_querygal']
                self.y = data['y_querygal']
                self.cam_ids = data['cam_ids']
            else:
                self.X = data['X_val']
                self.y = data['y_val']
        
        assert len(self.X) == len(self.y)
        self.train = train
        self.test = test
        
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        inp, target = self.X[idx], self.y[idx]
        if self.transform is not None:
            inp = self.transform(inp)
        return inp, target

    
class ConcatenateDataset(Dataset):
    """ Concatena múltiplos objetos da classe Dataset, na ordem em que forem fornecidos. 
        Considera que o array y de cada dataset j está no range [0, num_classes_j).
        Assim, os valores das classes dos datasets são alterados, para não haver 
        conflito de classes entre datasets. Ao array y do dataset j é somado o somatório 
        acumulado do número de classes dos datasets 0 a j-1. 
    """

    def __init__(self, *datasets):

        self._total_length = 0
        self.datasets_numclasses = []
        self.datasets = []

        for i, dataset in enumerate(datasets):
            assert isinstance(dataset, Dataset), "o dataset informado não é da classe Dataset: {}".format(type(dataset))
            self._total_length += len(dataset)
            self.datasets_numclasses.append(dataset.y.max()+1)
            self.datasets.append(dataset)

        self.classes_offsets = np.concatenate((np.array([0]), np.cumsum(self.datasets_numclasses)))


    @property
    def num_classes(self):
        return int(np.sum(self.datasets_numclasses))

    def __len__(self):
        return self._total_length

    def __getitem__(self, idx):

        for dataset_index, dataset in enumerate(self.datasets):
            if idx < len(dataset):
                break
            else:
                idx -= len(dataset)

        inp, target = dataset[idx]
        target += self.classes_offsets[dataset_index]
        return inp, target
