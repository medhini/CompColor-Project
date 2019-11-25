import os
import argparse

import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader

class ColorDataset(Dataset):
    def __init__(self, root, split='train'):
        self.root = root
        self.split = split

        self.files = {}
        self.files[self.split] = os.listdir(self.root+'/'+self.split)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, idx):
        """Returns gray scale and RGB images"""
        
        filename = os.path.join(self.root, self.split, self.files[self.split][idx])
        gray_img = Image.open(filename).convert('L')
        rgb_img = Image.open(filename)

        gray_img = np.asarray(gray_img, dtype=np.uint8)
        rgb_img = np.asarray(rgb_img, dtype=np.uint8)

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

    



