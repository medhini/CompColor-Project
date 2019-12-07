import argparse
import os

import colorgram
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageCms
from skimage import color
from torch.utils.data import DataLoader, Dataset

import random

def hue_shift(img):
    """RGB to LAB, cache L. RGB to HSV, shift H to get H^SV. H^SV to L^A^B^. L^A^B^ to LA^B^."""

    # Convert colourspace
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")

    #RGB to HSV
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    img_lab = ImageCms.applyTransform(img, rgb2lab)
    L = np.asarray(img_lab)[:,:,0]
    img_hsv = img.convert('HSV')

    #hue shift
    img_hsv = np.asarray(img_hsv).copy()
    h = img_hsv[:,:,0]
    h = (h+np.random.randint(0,255))%255
    img_hsv[:,:,0] = h

    #retain original luminance
    img_rgb = Image.fromarray(img_hsv, mode="HSV").convert('RGB')
    img_lab_t = np.asarray(ImageCms.applyTransform(img_rgb, rgb2lab)).copy()
    img_lab_t[:,:,0] = L

    #convert LAB to RGB  
    img_recolor = Image.fromarray(img_lab_t, mode="LAB")
    lab2rgb = ImageCms.buildTransformFromOpenProfiles(lab_p, srgb_p, "LAB", "RGB")
    img_recolor = ImageCms.applyTransform(img_recolor, lab2rgb)
    
    # img_recolor.save('recolor.jpg')

    return img_recolor


def gamma_shift(img):

    # Convert colourspace
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")

    #RGB to LAB
    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    img_lab = ImageCms.applyTransform(img, rgb2lab)
    L = np.asarray(img_lab)[:,:,0]
    A = np.asarray(img_lab)[:,:,1] / 110.0
    B = np.asarray(img_lab)[:,:,2] / 110.0

    new_a = (A+1)/2
    new_b = (B+1)/2
    gamma = random.uniform(0.5,1)

    new_a = new_a**gamma
    new_b = new_b**gamma

    new_a = (new_a - 1)/2
    new_b = (new_b - 1)/2

    img_lab_t = np.asarray(img_lab).copy()
    img_lab_t[:,:,1] = new_a * 110.0
    img_lab_t[:,:,2] = new_b * 110.0

    #convert LAB to RGB  
    img_recolor = Image.fromarray(img_lab_t, mode="LAB")
    lab2rgb = ImageCms.buildTransformFromOpenProfiles(lab_p, srgb_p, "LAB", "RGB")
    img_recolor = ImageCms.applyTransform(img_recolor, lab2rgb)
    
    # img_recolor.save('recolor.jpg')

    return img_recolor, gamma

def gamma_shift_ab(ab, gamma):  
    new_ab = (ab + 1)/2
    new_ab = new_ab**gamma
    new_ab = (new_ab - 1)/2

    return new_ab

class ColorDataset(Dataset):
    def __init__(self, root, split='train', transform=None, use_palette=False, shift_palette=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.use_palette = use_palette

        self.files = {}
        self.files[self.split] = os.listdir(self.root+'/'+self.split)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, idx):
        """Returns gray scale and RGB images"""

        filename = os.path.join(self.root, self.split, self.files[self.split][idx])
        rgb_img_orig = Image.open(filename).convert('RGB')

        if self.transform:
            rgb_img = self.transform(rgb_img_orig)
        
        
        rgb_img = np.array(rgb_img)
        lab_img = color.rgb2lab(rgb_img).astype(np.float32)
        lab_img = transforms.ToTensor()(lab_img)

        luma = lab_img[0:1, ...] / 100.0    # [0, 1]
        chroma = lab_img[1:3, ...] / 110.0  # [-1, 1]

        #extracting palette using 'https://github.com/obskyr/colorgram.py'. pip install colorgram.py
        if self.use_palette:
            rgb_img_small = rgb_img_orig.resize((56,56))
            colors = colorgram.extract(rgb_img_small, 6) #extracting 6 main colors
            palette1 = []
            for clr in colors:
                rgb = np.asarray(clr.rgb).reshape((1,1,3))
                rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
                lab = color.rgb2lab(rgb).astype(np.float32)
                lab = transforms.ToTensor()(lab)
                ab = lab[1:3, ...] / 110.0
                palette1.append(ab)

            while len(palette1) < 6:
                palette1.append(torch.zeros((2,1,1)))

            palette1 = torch.stack(palette1)

            if self.shift_palette:
                #new hue-shifted palette
                img_recolored = gamma_shift(rgb_img_orig)
                img_recolored_sml = img_recolored.resize((56,56))
                colors = colorgram.extract(img_recolored_sml, 6) #extracting 6 main colors

                palette = []
                for clr in colors:
                    rgb = np.asarray(clr.rgb).reshape((1,1,3))
                    rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
                    lab = color.rgb2lab(rgb).astype(np.float32)
                    lab = transforms.ToTensor()(lab)
                    ab = lab[1:3, ...] / 110.0
                    palette.append(ab)

                while len(palette) < 6:
                    palette.append(torch.zeros((2,1,1)))

                palette = torch.stack(palette)

                return luma, chroma, palette1, palette

            return luma, chroma, palette1
        else:
            return luma, chroma

class ReColorDataset(Dataset):
    def __init__(self, root, split='train', transform=None, use_palette=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.use_palette = use_palette

        self.files = {}
        self.files[self.split] = os.listdir(self.root+'/'+self.split)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, idx):
        """Returns gray scale and RGB images"""

        filename = os.path.join(self.root, self.split, self.files[self.split][idx])
        rgb_img_orig = Image.open(filename).convert('RGB')

        if self.transform:
            rgb_img = self.transform(rgb_img_orig)
        
        rgb_img = np.array(rgb_img)
        lab_img = color.rgb2lab(rgb_img).astype(np.float32)
        lab_img = transforms.ToTensor()(lab_img)

        luma = lab_img[0:1, ...] / 100.0    # [0, 1]
        chroma = lab_img[1:3, ...] / 110.0  # [-1, 1]

        img_recolored, gamma = gamma_shift(rgb_img_orig)
        
        if self.transform:
            img_recolored_t = self.transform(img_recolored)

        img_recolored_t = np.array(img_recolored_t)
        lab_img = color.rgb2lab(img_recolored_t).astype(np.float32)
        lab_img = transforms.ToTensor()(lab_img)

        chroma_shifted = lab_img[1:3, ...] / 110.0  # [-1, 1]

        #extracting palette using 'https://github.com/obskyr/colorgram.py'. pip install colorgram.py
        rgb_img_small = rgb_img_orig.resize((56,56))
        colors = colorgram.extract(rgb_img_small, 6) #extracting 6 main colors
        palette1 = []
        palette2 = []

        for clr in colors:
            rgb = np.asarray(clr.rgb).reshape((1,1,3))
            rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
            lab = color.rgb2lab(rgb).astype(np.float32)
            lab = transforms.ToTensor()(lab)
            ab = lab[1:3, ...] / 110.0
            ab_new = gamma_shift_ab(ab, gamma) 
            palette2.append(ab_new)
            palette1.append(ab)

        while len(palette1) < 6:
            palette1.append(torch.zeros((2,1,1)))
            palette2.append(torch.zeros((2,1,1)))

        palette1 = torch.stack(palette1)
        palette2 = torch.stack(palette2)

        #new shifted palette
        # img_recolored_sml = img_recolored.resize((56,56))
        # colors = colorgram.extract(img_recolored_sml, 6) #extracting 6 main colors

        # palette2 = []
        # for clr in colors:
        #     rgb = np.asarray(clr.rgb).reshape((1,1,3))
        #     rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
        #     lab = color.rgb2lab(rgb).astype(np.float32)
        #     lab = transforms.ToTensor()(lab)
        #     ab = lab[1:3, ...] / 110.0
        #     palette2.append(ab)

        # while len(palette2) < 6:
        #     palette2.append(torch.zeros((2,1,1)))

        # palette2 = torch.stack(palette2)

        return luma, chroma, chroma_shifted, palette1, palette2

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
