from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
import csv
from glob import glob
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class TaskData(Dataset):

    """
    this class will load data for a specific task, thus if the label of the task is missing for a case,
    that case will be omitted by the dataloader
    """

    def __init__(self,
                 Task,          # name of the task (column name)
                 task_config,   # task configuration
                 csv_dir,       # path for the csv table
                 stage,         # stage could be 'train' or 'valid' or 'test'
                 patch=None,    # patch could be 'random' location patch sample, 'fixed' location patch sample or None for whole volume reading
                 seed=1000,     # random seed
                 trans=None):   # data augmentation
        random.seed(seed)
        self.task = Task
        self.patch = patch
        self.sampler = PatchGenerator(47)
        self.trans = trans
        self.task_config = task_config
        self.Data_list, self.Labels_list, self.dataset_name_list = read_task_csv(csv_dir + '{}.csv'.format(stage), Task, task_config)

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        labels = self.Labels_list[idx]
        dataset_name = self.dataset_name_list[idx]
        if self.patch == 'random':
            data = np.load(self.Data_list[idx].replace('.nii', '.npy'), mmap_mode='r').astype(np.float32)
            patches = self.sampler.random_sample(data)
            return np.expand_dims(patches, axis=0), labels, dataset_name
        elif self.patch == 'fixed':
            data = np.load(self.Data_list[idx].replace('.nii', '.npy'), mmap_mode='r').astype(np.float32)
            patches = self.sampler.fixed_sample(data)
            return np.expand_dims(patches, axis=0), labels, dataset_name
        else:
            data = np.load(self.Data_list[idx].replace('.nii', '.npy')).astype(np.float32)
            if self.trans: data = self.trans.apply(data).astype(np.float32)
            data = np.expand_dims(data, axis=0)
            return data, labels, dataset_name

    def get_sample_weights(self, ratio={}):
        # ratio = {'PD':0.2} means averagely sample 2 PD cases and 8 no PD cases for each batch
        weights = []
        if self.task in ratio:
            for i in self.Labels_list:
                if i == 0:
                    weights.append(1-ratio[self.task])
                elif i == 1:
                    weights.append(ratio[self.task])
            return weights
        # if task is not in ratio, no specific value for the ratio, thus auto-balance the data according to data distribution
        unique = list(set(self.Labels_list))
        count = [float(self.Labels_list.count(a)) for a in unique]
        total = float(len(self.Labels_list))
        for i, name in zip(self.Labels_list, self.dataset_name_list):
            if 'ADNI' in name: name = 'ADNI'
            factor = self.task_config['sampleWeights'][name]
            unique_idx = unique.index(i)
            weights.append(total/count[unique_idx]*factor)
        return weights


class Test_Data(Dataset):
    """
    This class will load all cases from a csv file
    """
    def __init__(self, csv_file, padding=False):
        self.padding = padding
        self.Data_list = read_filenames(csv_file)
        self.Data_list = [a.replace('.nii', '.npy') for a in self.Data_list]

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        data = np.load(self.Data_list[idx]).astype(np.float32)
        if self.padding: data = self.pad(data)
        data = np.expand_dims(data, axis=0)
        return data, self.Data_list[idx]

    def pad(self, tensor, win_size=23):
        A = np.zeros((tensor.shape[0]+2*win_size, tensor.shape[1]+2*win_size, tensor.shape[2]+2*win_size))
        A[win_size:win_size+tensor.shape[0], win_size:win_size+tensor.shape[1], win_size:win_size+tensor.shape[2]] = tensor
        return A.astype(np.float32)


class PatchGenerator:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def random_sample(self, data):
        """sample random patch from numpy array data"""
        X, Y, Z = data.shape
        x = random.randint(0, X-self.patch_size)
        y = random.randint(0, Y-self.patch_size)
        z = random.randint(0, Z-self.patch_size)
        return data[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]

    def fixed_sample(self, data):
        """sample patch from fixed locations"""
        patches = []
        patch_locs = [[25, 90, 30], [115, 90, 30], [67, 90, 90], [67, 45, 60], [67, 135, 60]]
        for i, loc in enumerate(patch_locs):
            x, y, z = loc
            patch = data[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
            patches.append(np.expand_dims(patch, axis = 0))
        return patches


class Aug:
    def __init__(self):
        self.contrast_factor = 0.2
        self.bright_factor = 0.4
        self.sig_factor = 0.4

    def change_contrast(self, image):
        ratio = 1 + (random.random() - 0.5)*self.contrast_factor # ratio range is -0.9 to 1.1
        return image * ratio

    def change_brightness(self, image):
        val = (random.random() - 0.5)*self.bright_factor
        return image + val

    def add_noise(self, image):
        sig = random.random() * self.sig_factor
        return np.random.normal(0, sig, image.shape) + image

    def apply(self, image):
        image = self.change_contrast(image)
        image = self.change_brightness(image)
        image = self.add_noise(image)
        return image


def read_task_csv(filename, task, config):
    mri_list, label_list, dataset_name_list = [], [], []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row[task]: continue
            mri_list.append(row['path'] + row['filename'])
            dataset_name_list.append(map_path_to_dataset(row['path']))
            if config['type'] == 'reg':
                label_list.append(float(row[task]))
            elif config['type'] == 'cla':
                label_list.append(int(row[task]))
            else:
                raise NameError ('task type can only be either reg or cla')
    return mri_list, label_list, dataset_name_list


def read_filenames(filename):
    data_list = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if (row['path'] + row['filename']):
                data_list.append(row['path'] + row['filename'])
    return data_list


def padding(tensor, win_size=23):
    A = np.ones((tensor.shape[0]+2*win_size, tensor.shape[1]+2*win_size, tensor.shape[2]+2*win_size)) * tensor[-1,-1,-1]
    A[win_size:win_size+tensor.shape[0], win_size:win_size+tensor.shape[1], win_size:win_size+tensor.shape[2]] = tensor
    return A.astype(np.float32)


def map_path_to_dataset(path):
    for candi in ['ADNI', 'NACC', 'FHS', 'AIBL', 'OASIS', 'Stanford', 'PPMI', 'NIFD']:
        if candi in path:
            return candi
    return 'unknown'










