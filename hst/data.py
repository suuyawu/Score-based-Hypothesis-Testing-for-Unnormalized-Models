import copy
import os
import torch
import numpy as np
import datasets
import models
from config import cfg
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate


def fetch_dataset(data_name, params=None, verbose=True):
    dataset = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = os.path.join('data', data_name)
    if data_name in ['MVN']:
        dataset['test'] = datasets.MVN(root, **params)
    elif data_name in ['GMM']:
        dataset['test'] = datasets.GMM(root, **params)
    elif data_name in ['RBM']:
        dataset['test'] = datasets.RBM(root, **params)
    elif data_name in ['KDDCUP99']:
        dataset['train'] = datasets.KDDCUP99(root, 'train')
        dataset['test'] = datasets.KDDCUP99(root, 'test')
    elif data_name in ['EXP']:
        dataset['test'] = datasets.EXP(root, **params)
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        return {key: [b[key] for b in batch] for key in batch[0]}
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag, batch_size=None, shuffle=None, drop_last=None, sampler=None):
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        _drop_last = cfg[tag]['drop_last'][k] if drop_last is None else drop_last[k]
        if sampler is None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=True, num_workers=cfg['num_workers'], drop_last=_drop_last,
                                        collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                        pin_memory=True, num_workers=cfg['num_workers'], drop_last=_drop_last,
                                        collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
    return data_loader


def split_dataset(dataset):
    dataset_ = []
    for i in range(cfg['target_size']):
        dataset_i = copy.deepcopy(dataset)
        mask = dataset_i['test'].target == i
        dataset_i['test'].id = dataset_i['test'].id[mask]
        dataset_i['test'].data = dataset_i['test'].data[mask]
        dataset_i['test'].target = dataset_i['test'].target[mask]
        dataset_i['test'].id = dataset_i['test'].id[:cfg['num_samples']]
        dataset_i['test'].data = dataset_i['test'].data[:cfg['num_samples']]
        dataset_i['test'].target = dataset_i['test'].target[:cfg['num_samples']]
        dataset_.append(dataset_i)
    return dataset_
