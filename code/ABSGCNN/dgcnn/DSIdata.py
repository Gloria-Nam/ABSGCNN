from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import pandas as pd
import random
import os
from shutil import copy
from math import cos, sin, atan2, sqrt, pi ,radians, degrees
from pymatgen.core.structure import Structure
import collections

from dgcnn.dg_data import *

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

def get_weight_parameters(dataset):
    targets = []
    num = len(dataset)
    for i, (_, _, _, target, _) in enumerate(dataset): 
        targets.append(int(target)) 
    dict_targets = collections.Counter(targets)
    if dict_targets[0] != 0:
        dict_targets[0] = num/dict_targets[0]
    else:
        dict_targets[0] = 0 
    if dict_targets[1] != 0:        
        dict_targets[1] = num/dict_targets[1]
    else:
        dict_targets[1] = 0
    weight = [dict_targets[0] if int(dataset[i][-2]) == 0 else dict_targets[1] for i in range(num)]
    return weight, num
    
def DSIget_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if kwargs['train_size'] is None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                  f'test_ratio = {train_ratio} as training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    #回归采用SubsetRandomSampler，分类采用WeightedRandomSampler   
    if kwargs['task'] == 'regression':     
        train_sampler = SubsetRandomSampler(indices[:train_size])
        val_sampler = SubsetRandomSampler(
            indices[-(valid_size + test_size):-test_size])
        if return_test:
            test_sampler = SubsetRandomSampler(indices[-test_size:]) 
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn, pin_memory=pin_memory, drop_last=True)
        val_loader = DataLoader(dataset, batch_size=batch_size,
                                sampler=val_sampler,
                                num_workers=num_workers,
                                collate_fn=collate_fn, pin_memory=pin_memory, drop_last=True)
        if return_test:
            test_loader = DataLoader(dataset, batch_size=batch_size,
                                     sampler=test_sampler,
                                     num_workers=num_workers,
                                     collate_fn=collate_fn, pin_memory=pin_memory)            
    elif kwargs['task'] == 'classification':        
        train_val_set, test_set = random_split(dataset, [total_size - test_size, test_size])
        train_set, valid_set = random_split(train_val_set, [total_size - test_size - valid_size, valid_size])
        train_weight, train_num = get_weight_parameters(train_set)
        valid_weight, valid_num = get_weight_parameters(valid_set)
        test_weight, test_num = get_weight_parameters(test_set)
        train_sampler = WeightedRandomSampler(train_weight, train_num, replacement=True)
        val_sampler = WeightedRandomSampler(valid_weight, valid_num, replacement=True)
#         val_sampler = SubsetRandomSampler(list(range(len(valid_set))))
        if return_test:
#             test_sampler = WeightedRandomSampler(test_weight, test_num, replacement=True) 
            test_sampler = SubsetRandomSampler(list(range(len(test_set))))
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn, pin_memory=pin_memory, drop_last=True)
        val_loader = DataLoader(valid_set, batch_size=batch_size,
                                sampler=val_sampler,
                                num_workers=num_workers,
                                collate_fn=collate_fn, pin_memory=pin_memory, drop_last=True)
        if return_test:
            if len(test_set) % batch_size == 0:
                test_loader = DataLoader(test_set, batch_size=batch_size,
                                         sampler=test_sampler,
                                         num_workers=num_workers,
                                         collate_fn=collate_fn, pin_memory=pin_memory) 
            elif len(test_set) % batch_size != 0 and len(test_set) < 200:
                test_loader = DataLoader(test_set, batch_size=len(test_set),
                                         sampler=test_sampler,
                                         num_workers=num_workers,
                                         collate_fn=collate_fn, pin_memory=pin_memory) 
            elif len(test_set) % batch_size != 0 and len(test_set) > 200:
                test_loader = DataLoader(test_set, batch_size=batch_size,
                                         sampler=test_sampler,
                                         num_workers=num_workers,
                                         collate_fn=collate_fn, pin_memory=pin_memory, drop_last=True)                 
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def DSIcollate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea_d, batch_atom_fea_nbr_d, batch_nbr_fea_d, batch_nbr_fea_idx_d, batch_atom_fea_i, batch_nbr_fea_i, batch_nbr_fea_idx_i, batch_atom_fea_s, batch_nbr_fea_s, batch_nbr_fea_idx_s = [], [], [], [], [], [], [], [], [], []
    crystal_atom_idx_d, crystal_atom_idx_i, crystal_atom_idx_s, batch_target = [], [], [], []
    batch_cif_ids = []
    base_idx_d = 0
    base_idx_i = 0
    base_idx_s = 0
    for i, ((atom_fea_d, atom_fea_nbr_d, nbr_fea_d, nbr_fea_idx_d), (atom_fea_i, nbr_fea_i, nbr_fea_idx_i), (atom_fea_s, nbr_fea_s, nbr_fea_idx_s), target, cif_id)\
            in enumerate(dataset_list):
        # dot part
        n_d = atom_fea_d.shape[0]  # number of atoms for this single atom
        batch_atom_fea_d.append(atom_fea_d)
        batch_atom_fea_nbr_d.append(atom_fea_nbr_d)
        batch_nbr_fea_d.append(nbr_fea_d)
        batch_nbr_fea_idx_d.append(torch.where(nbr_fea_idx_d != -1, base_idx_d + nbr_fea_idx_d, nbr_fea_idx_d))# 不等于-1加base_idx_d 
        new_idx_d = torch.LongTensor(np.arange(n_d)+base_idx_d)
        crystal_atom_idx_d.append(new_idx_d)
        base_idx_d += n_d
        # inter part
        n_i = atom_fea_i.shape[0]  # number of atoms for this crystal
        batch_atom_fea_i.append(atom_fea_i)
        batch_nbr_fea_i.append(nbr_fea_i)
        batch_nbr_fea_idx_i.append(torch.where(nbr_fea_idx_i != -1, base_idx_i + nbr_fea_idx_i, nbr_fea_idx_i))       
        new_idx_i = torch.LongTensor(np.arange(n_i)+base_idx_i)
        crystal_atom_idx_i.append(new_idx_i)
        base_idx_i += n_i
        # surface part
        n_s = atom_fea_s.shape[0]  # number of atoms for this crystal
        batch_atom_fea_s.append(atom_fea_s)
        batch_nbr_fea_s.append(nbr_fea_s)
        batch_nbr_fea_idx_s.append(torch.where(nbr_fea_idx_s != -1, base_idx_s + nbr_fea_idx_s, nbr_fea_idx_s))
        new_idx_s = torch.LongTensor(np.arange(n_s)+base_idx_s)
        crystal_atom_idx_s.append(new_idx_s)
        base_idx_s += n_s
        # target part
        batch_target.append(target)
        batch_cif_ids.append(cif_id) 
#     print(torch.cat(batch_atom_fea_d, dim=0).shape, torch.cat(batch_atom_fea_nbr_d, dim=0).shape, torch.cat(batch_nbr_fea_d, dim=0).shape, torch.cat(batch_nbr_fea_idx_d, dim=0).shape)
    return (torch.cat(batch_atom_fea_d, dim=0),
            torch.cat(batch_atom_fea_nbr_d, dim=0),
            torch.cat(batch_nbr_fea_d, dim=0),
            torch.cat(batch_nbr_fea_idx_d, dim=0),
            crystal_atom_idx_d),\
          (torch.cat(batch_atom_fea_i, dim=0),
            torch.cat(batch_nbr_fea_i, dim=0),
            torch.cat(batch_nbr_fea_idx_i, dim=0),
            crystal_atom_idx_i),\
          (torch.cat(batch_atom_fea_s, dim=0),
            torch.cat(batch_nbr_fea_s, dim=0),
            torch.cat(batch_nbr_fea_idx_s, dim=0),
            crystal_atom_idx_s),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids


class DSIStructureData(Dataset): 
    """
        
        
    """
    def __init__(self, root_dir, structures_formula, layersNum_parameter, cut_rds=[8,8,8], nbr_type=['T', 'T', 'T'], random_seed=123):
        self.root_dir = root_dir
        self.structures_formula = structures_formula
        self.parameters = layersNum_parameter
        self.cut_rds = cut_rds
        self.nbr_type = nbr_type
        assert os.path.exists(root_dir), 'root_dir does not exist!' #判断为False时触发
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
         
        sample_id = [self.id_prop_data[i][0] for i in range(len(self.id_prop_data))]              
        training_dot_num, self.training_atom_d, self.training_atom_nbr_d, self.training_bond_d, self.training_bond_index1_d, self.training_bond_index2_d, training_atom_num_d, training_bond_num_d, training_inter_num, self.training_atom_i, self.training_bond_i, self.training_bond_index1_i, self.training_bond_index2_i, training_atom_num_i, training_bond_num_i, training_surface_num, self.training_atom_s, self.training_bond_s, self.training_bond_index1_s, self.training_bond_index2_s, training_atom_num_s, training_bond_num_s = data_preparation(sample_id, self.root_dir,     self.structures_formula, self.parameters, self.cut_rds, self.nbr_type)
        
    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]#获取id和目标值
        # dot part
        atom_fea_d = self.training_atom_d[idx] 
        atom_fea_nbr_d = self.training_atom_nbr_d[idx] 
        nbr_fea_d = self.training_bond_d[idx]
        nbr_fea_index_d = self.training_bond_index2_d[idx]      
        atom_fea_d = torch.Tensor(atom_fea_d)
        atom_fea_nbr_d = torch.Tensor(atom_fea_nbr_d)
        nbr_fea_d = torch.Tensor(nbr_fea_d)
        nbr_fea_index_d = torch.LongTensor(nbr_fea_index_d)
        
        # inter part
        atom_fea_i = self.training_atom_i[idx] 
        nbr_fea_i = self.training_bond_i[idx]
        nbr_fea_index_i = self.training_bond_index2_i[idx]      
        atom_fea_i = torch.Tensor(atom_fea_i)
        nbr_fea_i = torch.Tensor(nbr_fea_i)
        nbr_fea_index_i = torch.LongTensor(nbr_fea_index_i)
        
        # surface part        
        atom_fea_s = self.training_atom_s[idx]
        nbr_fea_s = self.training_bond_s[idx]
        nbr_fea_index_s = self.training_bond_index2_s[idx]
        atom_fea_s = torch.Tensor(atom_fea_s)
        nbr_fea_s = torch.Tensor(nbr_fea_s)
        nbr_fea_index_s = torch.LongTensor(nbr_fea_index_s)
              
        target = torch.Tensor([float(target)])
        return (atom_fea_d, atom_fea_nbr_d, nbr_fea_d, nbr_fea_index_d), (atom_fea_i, nbr_fea_i, nbr_fea_index_i), (atom_fea_s, nbr_fea_s, nbr_fea_index_s), target, cif_id        
