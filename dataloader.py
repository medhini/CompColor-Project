import os
import argparse

import torch
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

class CityScapesDataLoader(Dataset):
    def __init__(self, root, split="train"):
        self.root = root
        self.split = split
        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)

        self.files = {}
        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png") 

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, idx):
        img = Image.open(self.files[self.split][idx].rstrip())
        img = np.array(img, dtype=np.uint8)

        return img

class FlickrDataLoader(Dataset):
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

    # trainset = CityScapesDataLoader('/shared/medhini/data/cityscapes/')
    # trainloader = DataLoader(trainset, batch_size=1, num_workers=0)

    # for i,data in enumerate(trainloader, 0):
    #     image = np.asarray(data[0])
    #     image = Image.fromarray(np.uint8(image), 'RGB')
    #     image.save('trial.jpg')
    #     break

    trainset = FlickrDataLoader('/tmp_data1/flickr30k-images')
    trainloader = DataLoader(trainset, batch_size=1, num_workers=0)

    for i,(gray_img, rgb_img) in enumerate(trainloader, 0):
        print(gray_img.shape, rgb_img.shape)
        
        image = Image.fromarray(np.uint8(gray_img[0]), 'L')
        image.save('trial.jpg')

        image = Image.fromarray(np.uint8(rgb_img[0]), 'RGB')
        image.save('rgb_trial.jpg')
        break

    



