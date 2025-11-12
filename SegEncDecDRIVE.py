import os
import numpy as np
import glob
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim

from DRIVEDataset import DRIVEDataset
from PH2Dataset import PH2Dataset

from losses import *
from seg_train import *
from custom_models import *


### Load Data ###
train_split = 0.8
batch_size = 1
size = (128,128)#(256,256)#(128,128)(282,292)
transform = transforms.Compose([transforms.Resize(size), 
                                    transforms.ToTensor()])

drive_data = DRIVEDataset(transform)
generator = torch.Generator().manual_seed(42)

drive_train, drive_test, drive_val = random_split(drive_data,
                                       [train_split,(1-train_split)/2,(1-train_split)/2],
                                       generator=generator)

drive_loader_train = DataLoader(drive_train, batch_size=batch_size, shuffle=True)
drive_loader_test = DataLoader(drive_test, batch_size=batch_size, shuffle=False)
drive_loader_val = DataLoader(drive_val, batch_size=batch_size, shuffle=False)

### Configure device ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

## Configure Model ###
encdec = EncDec()
encdec.to(device)
learning_rate = 0.0001
opt = optim.Adam(encdec.parameters(), lr=learning_rate)

epochs = 160

log = train_seg(encdec,
                optimizer=opt,
                loss_fn=DiceLoss(),
                train_loader=drive_loader_train,
                test_loader=drive_loader_test,
                val_loader=drive_loader_val,
                num_epochs=epochs
                )


with open('seg_log.json', 'r') as f:
    log_data = json.load(f)

# Plot and save
plot_segmentation_metrics(log_data)