import torch
import torchvision.transforms.functional as AF
import torch.nn.functional as F
import nibabel as nib
import torchvision.transforms as T

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Cutoff values of HU (Lung window)
HU_min = -1000
HU_max = 400



# Function to augment data
def data_augmentation(image, mask1, mask2):
    # Horizontal flip
    if torch.rand(1) > 0.5:
        image = AF.hflip(image)
        mask1 = AF.hflip(mask1)
        mask2 = AF.hflip(mask2)

    # Vertical flip
    if torch.rand(1) > 0.5:
        image = AF.vflip(image)
        mask1 = AF.vflip(mask1)
        mask2 = AF.vflip(mask2)
    # Rotate 90 degree or -90 degree
    if torch.rand(1) > 0.5:
        if torch.rand(1) > 0.5:
            image = torch.rot90(image, 1, [2, 3])
            mask1 = torch.rot90(mask1, 1, [2, 3])
            mask2 = torch.rot90(mask2, 1, [2, 3])
        else:
            image = torch.rot90(image, -1, [2, 3])
            mask1 = torch.rot90(mask1, -1, [2, 3])
            mask2 = torch.rot90(mask2, -1, [2, 3])

    return image, mask1, mask2

# Focal Tversky Loss
def FocalTverskyLoss(inputs, targets, smooth=1, alpha=0.8, beta=0.2, gamma=(4 / 3), device=device):
    inputs = torch.unsqueeze(inputs, dim=1)
    targets = torch.unsqueeze(targets, dim=1)

    inputs = inputs.to(device)
    targets = targets.to(device)

    # comment out if model contains a sigmoid or equivalent activation layer
    # inputs = torch.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.contiguous().view(-1)
    targets = targets.contiguous().view(-1)

    # True Positives, False Positives & False Negatives
    TP = (inputs * targets).sum()
    FP = ((1-targets) * inputs).sum()
    FN = (targets * (1-inputs)).sum()

    Tversky = (TP+smooth) / (TP+(alpha * FP)+(beta * FN)+smooth)

    FocalTversky = (1-Tversky) ** gamma

    return FocalTversky

# Dice Score
def DCE(inputs, targets, smooth=1):
    # inputs = F.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection+smooth) / (inputs.sum()+targets.sum()+smooth)
    return dice

# combine label to a single tensor. Add background also. Background is anything other than lungs and lesion
def label_rearrange(lung_or_lesion):
    target = torch.zeros((lung_or_lesion.shape[0], 2, 256, 256))
    backgrd=lung_or_lesion.squeeze() + 1
    backgrd[backgrd==2]=0
    target[:, 0, :, :] = lung_or_lesion.squeeze()
    target[:,1,:,:]=backgrd
    return target

# Calculate loss
def loss_calc(out, pred1,pred2, lab_out, lab1,lab2):
    loss_bce = torch.nn.BCEWithLogitsLoss()
    loss1 = FocalTverskyLoss(pred1[:,0,:,:], lab1[:,0,:,:])
    loss2 = FocalTverskyLoss(pred2[:,0,:,:], lab2[:,0,:,:])
    loss3 = loss_bce(out, lab_out)
    loss = 0.35*loss1+0.35*loss2+0.3*loss3
    return loss,loss1,loss2,loss3


# Calculate loss
def loss_calc_test(out, pred1,pred2, lab_out, lab1,lab2):
    loss_bce = torch.nn.BCEWithLogitsLoss()
    loss1 = DCE(pred1[:,0,:,:], lab1[:,0,:,:])
    loss2 = DCE(pred2[:,0,:,:], lab2[:,0,:,:])
    loss3 = loss_bce(out, lab_out)
    loss = 0.35*loss1+0.35*loss2+0.3*loss3
    return loss,loss1,loss2,loss3


# One hot encode the classification labels
def change_lab(tar1,tar2):
    targ=torch.zeros((tar1.shape[0], 2, 256, 256))
    targ[:,0,:,:]=tar1[:,0,:,:]
    targ[:,1,:,:]=tar2[:,0,:,:]
    target=torch.sum(targ,dim=1)
    class_lab1 = torch.max(target, dim=1)
    class_lab2 = torch.max(class_lab1.values, dim=1)
    lab = class_lab2.values
    class_lab = F.one_hot(lab.long(), num_classes=3)

    return class_lab

def data_preprocess(path, hu_min, hu_max):
    volume_data = nib.load(path)  # load data
    volume_data_numpy = volume_data.get_fdata()  # get data as numpy
    volume_data_tensor = torch.tensor(volume_data_numpy)  # convert to torch tensor
    if torch.max(volume_data_tensor) > 300:
        volume_data_tensor_clamped = torch.clamp(volume_data_tensor, min=hu_min, max=hu_max)  # apply HU lung window
    else:
        hu_min = 0
        hu_max = 200
        volume_data_tensor_clamped = torch.clamp(volume_data_tensor, min=hu_min, max=hu_max)  # apply HU lung window
    volume_data_tensor_clamped_normalized = (volume_data_tensor_clamped-hu_min) / (hu_max-hu_min)  # normalize to [0,1]

    return volume_data_tensor_clamped_normalized


# function to obtain maask
def mask_obtain(fpath):
    mask = nib.load(fpath)  # load mask
    mask_numpy = mask.get_fdata()  # get mask as numpy
    mask_tensor = torch.tensor(mask_numpy)  # convert to torch tensor

    return mask_tensor

# resize the CT scans
def vol_resize(data,inp_resize_val):
    if data.shape[1] != inp_resize_val or data.shape[2] != inp_resize_val:
        data = T.Resize([inp_resize_val, inp_resize_val])(data)
    return data

# obtain data and masks as tensors
def get_data(inp_resize_val, data_numbers, file_name, data_path, mask_lung_path,mask_lesion_path):
    out_data = torch.empty((1, inp_resize_val, inp_resize_val))
    out_lung_mask = torch.empty((1, inp_resize_val, inp_resize_val))
    out_lesion_mask = torch.empty((1, inp_resize_val, inp_resize_val))

    for i in data_numbers:
        file_path = data_path+file_name[i]  # path of the data
        data = data_preprocess(file_path, HU_min, HU_max)  # preprocess data
        data = data.permute(2, 0, 1)  # change the dimension (H,W,C) ---> (C,H,W)
        data = vol_resize(data,inp_resize_val)
        out_data = torch.cat((out_data, data), 0)  # stack the data(slices) along dimension C

        lung_mask_file_path = mask_lung_path+file_name[i]  # path of the mask
        lung_mask = mask_obtain(lung_mask_file_path)  # preprocess mask
        lung_mask = lung_mask.permute(2, 0, 1)  # change the dimension (H,W,C) ---> (C,H,W)
        lung_mask = vol_resize(lung_mask,inp_resize_val)
        lung_mask_index = (lung_mask > 0).nonzero(as_tuple=True)
        lung_mask[lung_mask_index] = 1
        out_lung_mask = torch.cat((out_lung_mask, lung_mask), 0)  # stack the masks along dimension C

        lesion_mask_file_path = mask_lesion_path+file_name[i]  # path of the mask
        lesion_mask = mask_obtain(lesion_mask_file_path)  # preprocess mask
        lesion_mask = lesion_mask.permute(2, 0, 1)  # change the dimension (H,W,C) ---> (C,H,W)
        lesion_mask = vol_resize(lesion_mask,inp_resize_val)
        lesion_mask_index = (lesion_mask > 0).nonzero(as_tuple=True)
        lesion_mask[lesion_mask_index] = 1
        out_lesion_mask = torch.cat((out_lesion_mask, lesion_mask), 0)  # stack the masks along dimension C

    # remove the initial zero utilized to initialize the tensors
    out_dataset = out_data[1:, :, :]
    out_lesion_maskset = out_lesion_mask[1:, :, :]
    out_lung_maskset = out_lung_mask[1:, :, :]

    # Calculate the labels by finding the maximum value of the masks.
    labels_lesion = torch.amax(out_lesion_maskset, (1, 2))
    labels_lesion = labels_lesion+1 # change label to 1 and 2 instead of 0 and 1 for convenience in the sampler

    labels_lung = torch.amax(out_lung_maskset, (1, 2))
    labels_lung = labels_lung+1


    return out_dataset, out_lung_maskset, labels_lung,out_lesion_maskset,labels_lesion

