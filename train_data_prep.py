import os
import torch
from utils import get_data
import numpy as np
from torch.utils.data import Sampler, BatchSampler
from torch.utils.data import DataLoader, Dataset
import argparse
import torchvision.transforms.functional as AF

# Data and mask path
data_path = ".../Data/"
mask_lung_path = ".../Lung Mask/"
mask_lesion_path = ".../Mask/"


# names of files in data folder
arr = os.listdir(data_path)

# select train subjects
train_data_numbers = range(0, 20,2)


class EqualSampler(Sampler):
    def __init__(self,
                 lung_labels: None,
                 lesion_labels:None,
                 batches_per_epoch: int = None,
                 n: int = None,
                 k: int = None):
        self.batches_per_epoch = batches_per_epoch
        self.lung_labels = lung_labels
        self.lesion_labels=lesion_labels
        self.k = k
        self.n = n


    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        s1 = (self.lung_labels == 1).nonzero(as_tuple=False) # provides indices of CT slices which have closed clungs
        s2 = torch.tensor(np.intersect1d((self.lung_labels == 2).nonzero(as_tuple=False),(self.lesion_labels == 1).nonzero(as_tuple=False))) # indices of slices with normal open lung
        s3 = (self.lesion_labels == 2).nonzero(as_tuple=False) # indices of slices with lesions


        indices = []
        for _ in range(self.batches_per_epoch):
            # Permute the indices to randomize the data for each class
            s1_perm = s1[torch.randperm(s1.size()[0])]
            s2_perm = s2[torch.randperm(s2.size()[0])]
            s3_perm = s3[torch.randperm(s3.size()[0])]
            # Select k number of slices for each class
            img_idx1 = s1_perm[:self.k]
            img_idx2 = s2_perm[:self.k]
            img_idx3 = s3_perm[:self.k]
            img_idx = [img_idx1, img_idx2,img_idx3]
            # Obtain the list of indices for a single batch
            img_idx_single = [item for sublist in img_idx for item in sublist]
            indices.append([img_idx_single])

        return iter(indices)


batches_per_epoch = 100
import sys

sys.argv = ['']
del sys
parser = argparse.ArgumentParser()
parser.add_argument('--n-train', default=3, type=int) # number of classes
parser.add_argument('--n-test', default=3, type=int)
parser.add_argument('--k-train', default=10, type=int)# number of samples to be taken per class per batch
parser.add_argument('--k-test', default=10, type=int)

args = parser.parse_args()

inp_resize_val = 256 # resize value for the image


# Dataset Module
class CovData(Dataset):
    def __init__(self, data, lung_mask,lesion_mask):
        self.data = data
        self.lung_mask = lung_mask
        self.lesion_mask=lesion_mask


    def __len__(self):
        return self.lung_mask.size()[2]

    def __getitem__(self, item):

        support_slices = self.data[item[0], :, :]
        support_lung_masks = self.lung_mask[item[0], :, :]
        support_lesion_masks = self.lesion_mask[item[0], :, :]

        return support_slices, support_lung_masks, support_lesion_masks

# Obtain the entire training data as tensors
train_dataset, train_lung_maskset, train_lung_labels, train_lesion_maskset,train_lesion_labels = get_data(inp_resize_val, train_data_numbers, arr, data_path, mask_lung_path,mask_lesion_path)

# Contrast adjust the images
train_dataset=AF.autocontrast(torch.unsqueeze(train_dataset,1))
train_dataset=torch.squeeze(train_dataset)


# Obtain training dataset and dataloader
train_dataset = CovData(train_dataset, train_lung_maskset, train_lesion_maskset)
train_taskloader = DataLoader(train_dataset,
    batch_sampler=BatchSampler(EqualSampler(train_lung_labels,train_lesion_labels, batches_per_epoch, args.n_train, args.k_train), drop_last=False),
    num_workers=1)


