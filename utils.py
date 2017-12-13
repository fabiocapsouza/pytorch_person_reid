# -*- coding: utf-8 -*-

import numpy as np
import torch
import os.path as osp
import shutil

class MetricAverager(object):
    
    def __init__(self):
        self._n = 0
        self._mean = 0
        self._min = np.Inf
        
    def update(self, val):
        self._n += 1
        self._mean = (self._mean*(self._n - 1) + val) / self._n
        if val < self._min:
            self._min = val

    def clear(self):
        self._n = 0
        self._mean = 0
        self._min = np.Inf

    @property
    def mean(self):
        return self._mean

    @property
    def min(self):
        return self._min

    def __str__ (self):
        return 'MetricAverager(Mean: {:.5f}, n: {})'.format(self._mean, self._n)
        
    def __format__(self, format_spec):
        return format(self._mean, format_spec)



def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        fname, ext = osp.splitext(filename)
        best_filename = '{}_best{}'.format(*osp.splitext(filename))
        shutil.copyfile(filename, best_filename)


def load_checkpoint(filename, model, optimizer):
    print('Carregando modelo {}'.format(filename))
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    scheduler = checkpoint.get('scheduler', None)
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']
    train_history = checkpoint.get('history', [])
    
    print('Carregado.')
    return scheduler, epoch, best_val_loss, train_history