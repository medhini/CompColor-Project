import os
import argparse

import torch
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

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
        rgb_img = Image.open(filename).convert('L')

        if self.transform:
            rgb_img = self.transform(rgb_img)
        
        gray_img = rgb_img.convert('L')

        gray_img = torch.from_numpy(np.asarray(gray_img, dtype=np.int64))
        rgb_img = torch.from_numpy(np.asarray(rgb_img, dtype=np.int64))

        return gray_img, rgb_img

if __name__=='__main__':
    
    trainset = ColorDataset('/tmp_data1/flickr30k-images')
    trainloader = DataLoader(trainset, batch_size=1, num_workers=0)

    for i,(gray_img, rgb_img) in enumerate(trainloader, 0):
        print(gray_img.shape, rgb_img.shape)
        
        image = Image.fromarray(np.uint8(gray_img[0]), 'L')
        image.save('trial.jpg')

        image = Image.fromarray(np.uint8(rgb_img[0]), 'RGB')
        image.save('rgb_trial.jpg')
        break

    



