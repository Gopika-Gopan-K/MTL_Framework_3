import os
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_data
import torchvision.transforms.functional as AF


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Data and mask path
data_path = ".../Data/"
mask_lung_path = ".../Lung Mask/"
mask_lesion_path = ".../Mask/"


# names of files in data folder
arr = os.listdir(data_path)

# select test subjects
test_data_numbers = range(1,20,2)

inp_resize_val = 256 # image resize value


class CovData(Dataset):
    def __init__(self, data, lung_mask,lesion_mask):
        self.data = data
        self.lung_mask = lung_mask
        self.lesion_mask=lesion_mask


    def __len__(self):
        return self.lung_mask.size()[0]

    def __getitem__(self, item):

        support_slices = self.data[item, :, :]
        support_lung_masks = self.lung_mask[item, :, :]
        support_lesion_masks = self.lesion_mask[item, :, :]

        return support_slices, support_lung_masks, support_lesion_masks



test_dataset, test_lung_maskset, test_lung_labels, test_lesion_maskset, test_lesion_labels = get_data(inp_resize_val, test_data_numbers, arr, data_path, mask_lung_path,mask_lesion_path)
# Contrast adjust the images
test_dataset=AF.autocontrast(torch.unsqueeze(test_dataset,1))
test_dataset=torch.squeeze(test_dataset)

test_dataset1 = CovData(test_dataset, test_lung_maskset,test_lesion_maskset)
test_taskloader = DataLoader(test_dataset1,batch_size=1)



