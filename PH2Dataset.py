import torch
import os
import glob
import PIL.Image as Image

DATA_PATH = '/dtu/datasets1/02516/PH2_Dataset_images/'

class PH2Dataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        'Initialization'
        self.transform = transform
        #data_path = os.path.join(DATA_PATH, 'train' if train else 'test')
        # Each image is found in a path formed by IMDxxx, which inside contains a folder named IMDxxx_Dermoscopic_Image and IMDxxx_lesion, so we need to access each folder and get the path to the .bmp files inside
        self.image_paths = []
        self.label_paths = []
        for folder in glob.glob(DATA_PATH + 'IMD*/'):
            self.image_paths.extend(glob.glob(folder + 'IMD*_Dermoscopic_Image/*.bmp'))
            self.label_paths.extend(glob.glob(folder + 'IMD*_lesion/*.bmp'))

        self.image_paths = sorted(self.image_paths)
        self.label_paths = sorted(self.label_paths)

        assert len(self.image_paths) == len(self.label_paths), "Mismatch between images and labels"
        print(f"Number of loaded images: {len(self.image_paths)}, Number of loaded labels: {len(self.label_paths)}")
        print(f"Original image size: {Image.open(self.image_paths[0]).size}, Original label size: {Image.open(self.label_paths[0]).size}")
        print(f"Number of channels in image: {len(Image.open(self.image_paths[0]).getbands())}, Number of channels in label: {len(Image.open(self.label_paths[0]).getbands())}\n")

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
