import os
import numpy as np
import glob
import PIL.Image as Image
import json

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
batch_size = 4
size = (286,378) # (512,512)NOTE: The dataset prints W,H nor H,W!!!
transform = transforms.Compose([transforms.Resize(size), 
                                    transforms.ToTensor()])

ph2_data = PH2Dataset(transform)
generator = torch.Generator().manual_seed(42)

ph2_train, ph2_test, ph2_val = random_split(ph2_data,
                                       [train_split,(1-train_split)/2,(1-train_split)/2],
                                       generator=generator)

ph2_loader_train = DataLoader(ph2_train, batch_size=batch_size, shuffle=True)
ph2_loader_test = DataLoader(ph2_test, batch_size=batch_size, shuffle=False)
ph2_loader_val = DataLoader(ph2_val, batch_size=batch_size, shuffle=False)


### Configure device ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

### Load Model and Optimizer

encdec = PH2UNet()
encdec.to(device)
learning_rate = 0.001
opt = optim.Adam(encdec.parameters(), lr=learning_rate)

epochs = 16
log = train_seg(encdec,
                optimizer=opt,
                loss_fn=BCELoss(),
                train_loader=ph2_loader_train,
                test_loader=ph2_loader_test,
                val_loader=ph2_loader_val,
                num_epochs=epochs
                )
"""
log = train_seg_weak_supervision(encdec,
                                optimizer=opt,
                                num_pos_clicks=3,
                                num_neg_clicks=6,
                                train_loader=ph2_loader_train,
                                test_loader=ph2_loader_test,
                                num_epochs=epochs
                                )
"""


with open('seg_log.json', 'r') as f:
    log_data = json.load(f)

# Plot and save
plot_segmentation_metrics(log_data)

            