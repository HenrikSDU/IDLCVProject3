import torch
import os
import glob
import PIL.Image as Image

DATA_PATH = '/dtu/datasets1/02516/DRIVE'

class DRIVEDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        'Initialization'
        self.transform = transform
        
        self.image_paths = sorted(glob.glob(DATA_PATH + '/training' + '/images/*.tif'))
        self.label_paths = sorted(glob.glob(DATA_PATH + '/training' +'/1st_manual/*.gif'))
        
        assert len(self.image_paths) == len(self.label_paths), "Mismatch between images and labels"
        print(f"Number of loaded images: {len(self.image_paths)}, Number of loaded labels: {len(self.label_paths)}")
        print(f"Original image size: {Image.open(self.image_paths[0]).size}, Original label size: {Image.open(self.label_paths[0]).size}")
        print(f"Number of channels in image: {len(Image.open(self.image_paths[0]).getbands())}, Number of channels in label: {len(Image.open(self.label_paths[0]).getbands())}")

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y
