import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage import color
from torch.utils.data import DataLoader, Dataset


class ColorDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.files = {}
        self.files[self.split] = os.listdir(self.root+'/'+self.split)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, idx):
        """Returns gray scale and RGB images"""

        filename = os.path.join(self.root, self.split, self.files[self.split][idx])
        rgb_img = Image.open(filename).convert('RGB')

        if self.transform:
            rgb_img = self.transform(rgb_img)

        rgb_img = np.asarray(rgb_img) / 255.0
        # TODO: We need the input to be in LAB space, but it is RGB. Adding the
        # line below causes a sync issue with pytorch for some reason?
        lab_img = rgb_img  # color.rgb2lab(rgb_img)
        lab_img = torch.from_numpy(lab_img).permute(2, 0, 1).float()
        gray_img = lab_img[0:1, ...]

        return gray_img, lab_img

if __name__=='__main__':

    trainset = ColorDataset('/shared/timbrooks/datasets/mirflickr')
    trainloader = DataLoader(trainset, batch_size=8, num_workers=0)

    for i,(gray_img, rgb_img) in enumerate(trainloader, 0):
        print(gray_img.shape, rgb_img.shape)

        image = Image.fromarray(np.uint8(gray_img[0]), 'L')
        image.save('trial.jpg')

        image = Image.fromarray(np.uint8(rgb_img[0]), 'RGB')
        image.save('rgb_trial.jpg')
        break
