from train_data_prep import train_taskloader
from utils import data_augmentation, label_rearrange,change_lab,loss_calc
from model import mymodel
import torch
from tqdm import tqdm
import einops
import os

from torch.cuda import amp



os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Create GradScaler for mixed precision training
scaler = amp.GradScaler()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = mymodel()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.01)
num_epochs = 700

model.train()  # model in training mode

for epoch in range(num_epochs):
    dataset_size = 0
    running_loss = 0.0
    loop = tqdm(enumerate(train_taskloader), total=len(train_taskloader), leave=False)
    for batch_idx, (images, lung_lab, lesion_lab) in loop:
        torch.cuda.empty_cache()
        images, lung_lab, lesion_lab = data_augmentation(images, lung_lab, lesion_lab)

        # Rearrange to BCHW format
        if images.size()[0] == 1:
            images = einops.rearrange(images, 'c b h w -> b c h w')
            lung_lab = einops.rearrange(lung_lab, 'c b h w -> b c h w')
            lesion_lab = einops.rearrange(lesion_lab, 'c b h w -> b c h w')

        batch_size = images.size()[0]
        target_seg1 = label_rearrange(lung_lab)
        target_seg2 = label_rearrange(lesion_lab)
        target_class = change_lab(lung_lab, lesion_lab)

        images = images.to(device).type(torch.cuda.FloatTensor)
        target_seg1 = target_seg1.to(device).type(torch.cuda.FloatTensor)
        target_seg2 = target_seg2.to(device).type(torch.cuda.FloatTensor)
        target_class = target_class.to(device).type(torch.cuda.FloatTensor)

        optimizer.zero_grad()

        with amp.autocast():
            class_out, seg_out1,seg_out2 = model(images)
            loss,loss_seg1,loss_seg2,loss_clas = loss_calc(class_out, seg_out1,seg_out2, target_class, target_seg1,target_seg2)

        scaler.scale(loss).backward()  # scales loss and create scaled gradients for MPT
        # unscale the gradients of the optimizer assigned params, skips optimizer.step if Nan or Inf present
        scaler.step(optimizer)
        scaler.update()  # update scale for next iteration
        # Epoch loss calculation
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        # Update progress bar
        loop.set_description(f"Epoch : [{epoch}/{num_epochs}]")
        loop.set_postfix(loss=loss.item(),loss_seg1=loss_seg1.item(),loss_seg2=loss_seg2.item(),loss_seg3=loss_seg3.item(),loss_clas=loss_clas.item())

print(epoch_loss)
torch.save(model, f'model_framework3.pth')
