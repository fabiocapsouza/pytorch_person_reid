# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import Dataset


class Market1501(Dataset):
    def __init__(self, fpath, train=True, test=False, transform=None):

        with np.load(fpath) as data:
            #print(data.files)
            if train:
                self.X = data['X_train']   # (N,H,W,C)
                self.y = data['y_train']
            elif test:
                self.X_querygal = data['X_querygal']
                self.y_querygal = data['y_querygal']
                self.cam_ids_querygal = data['query_cam_ids']
                self.X_distractors = data['X_distractors']
                self.y_distractors = data['y_distractors']
                self.cam_ids_distractors = data['distractors_cam_ids']
            else:
                self.X = data['X_val']
                self.y = data['y_val']
                self.cam_ids = data['val_cam_ids']

        if train == False and test == True:
            self.mode = 'query'
        
        assert len(self.X) == len(self.y)
        self.train = train
        self.test = test
        
        self.transform = transform


    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in ['query', 'distractors']:
            raise ValueError('Mode must be either "query" or "distractors".')

        self._mode = value
        if value == 'query':
            self.X = self.X_querygal
            self.y = self.y_querygal
            self.cam_ids = self.cam_ids_querygal
        elif value == 'distractors':
            self.X = self.X_distractors
            self.y = self.y_distractors
            self.cam_ids = self.cam_ids_distractors
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        inp, target = self.X[idx], self.y[idx]
        if self.transform is not None:
            inp = self.transform(inp)
        return inp, target


class Market1501_query(Dataset):
    def __init__(self, fpath, transform=None):
        with np.load(fpath) as data:
            self.X_query = data['X_query']
            self.y_query = data['y_query']
            self.cam_ids = data['cam_ids']
            self.X_distractors = data['X_distractors']
            self.y_distractors = data['y_distractors']

        self.cameras = np.unique(self.cam_ids)
        self.pids = np.unique(self.y_query)
        
        assert len(self.X_query) == len(self.y_query)

        self.mode = 'query'

        self.transform = transform


    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in ['query', 'distractors']:
            raise ValueError('Mode must be either "query" or "distractors".')

        self._mode = value
        if value == 'query':
            self.X = self.X_query
            self.y = self.y_query
        elif value == 'distractors':
            self.X = self.X_distractors
            self.y = self.y_query


    def get_query_images_for_id(self, pid, return_all=False):
        ''' Returns one random image of the person with person ID `pid` of each camera. '''

        # Finds the indexes of the images belonging to that pid.
        # Then, selects the images and the camera ids for those images
        idx_person = np.argwhere(self.y_query == pid).squeeze()
        imgs = self.X_query[idx_person]
        cam_ids = self.cam_ids[idx_person]

        out_imgs = []
        out_cam_ids = []
        for c in self.cameras:
            c_imgs = imgs[cam_ids == c]
            if len(c_imgs) > 0:
                if return_all:
                    out_imgs.append(c_imgs)
                    out_cam_ids.extend([c]*len(c_imgs))
                else:
                    random_index = np.random.choice(np.arange(len(c_imgs)))
                    out_imgs.append(c_imgs[random_index])
                    out_cam_ids.append(c)

        return np.array(out_imgs), np.array(out_cam_ids)


    def get_query_indexes_for_id(self, pid, return_all=False):
        ''' Returns one random image of the person with person ID `pid` of each camera. '''

        # Finds the indexes of the images belonging to that pid.
        # Then, selects the images and the camera ids for those images
        idxs_person = np.argwhere(self.y_query == pid).squeeze()
        cam_ids = self.cam_ids[idxs_person]

        out_indexes = []
        out_cam_ids = []
        for c in self.cameras:
            c_idx = idxs_person[cam_ids == c]
            if len(c_idx) > 0:
                if return_all:
                    out_indexes.append(c_idx)
                    out_cam_ids.extend([c]*len(c_idx))
                else:
                    random_index = np.random.choice(np.arange(len(c_idx)))
                    out_indexes.append(c_idx[random_index])
                    out_cam_ids.append(c)

        return np.array(out_indexes), np.array(out_cam_ids)



    def get_gallery_indexes_for_query_input(self, idx):
        if self.mode != 'query':
            raise RuntimeError('Dataset is in "distractors" mode. It must be set to "query" mode.')

        cam_id = self.cam_ids[idx]
        return np.argwhere(self.cam_ids != cam_id).squeeze()




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
            elif test:
                self.X = data['X_querygal']
                self.y = data['y_querygal']
                self.cam_ids = data['cam_ids']
            else:
                self.X = data['X_val']
                self.y = data['y_val']
                self.cam_ids = data['val_cam_ids']
        
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


class VIPeR(Dataset):
    def __init__(self, fpath, train=True, test=False, transform=None):
        with np.load(fpath) as data:
            if train:
                self.X = data['X_train']   # (N,H,W,C)
                self.y = data['y_train']
                self.cam_ids = data['cam_ids_train']
            elif test:
                self.X = data['X_querygal']
                self.y = data['y_querygal']
                self.cam_ids = data['cam_ids_querygal']
            else:
                self.X = data['X_val']
                self.y = data['y_val']
                self.cam_ids = data['cam_ids_val']
        
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


class CUHK01(Dataset):
    def __init__(self, fpath, train=True, test=False, transform=None):
        with np.load(fpath) as data:
            if train:
                self.X = data['X_train']   # (N,H,W,C)
                self.y = data['y_train']
                self.cam_ids = data['cam_ids_train']
            elif test:
                self.X = data['X_querygal']
                self.y = data['y_querygal']
                self.cam_ids = data['cam_ids_querygal']
            else:
                self.X = data['X_val']
                self.y = data['y_val']
                self.cam_ids = data['cam_ids_val']
        
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

    @property
    def cam_ids(self):
        return np.concatenate([d.cam_ids for d in self.datasets], axis=0)

    @property
    def X(self):
        return np.concatenate([d.X for d in self.datasets], axis=0)

    @property
    def y(self):
        return np.concatenate([d.y+self.classes_offsets[i] for i,d in enumerate(self.datasets)], axis=0)

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
