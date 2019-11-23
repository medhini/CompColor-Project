import pickle
from PIL import Image
import numpy as np

import torch
import torchvision

def extract_pallete(img):

    return 

# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict

# def load_images():
#     N = 5
#     file_names = ['./data/cifar-10-batches-py/data_batch_%s'%(i+1) for i in range(N)]

#     for file in file_names:
#         image_batch = unpickle(file)
#         print(image_batch.keys())
#         images = image_batch['data']
#         print(len(images))

#     return

def hue_shift(img):
    """RGB to LAB, cache L. RGB to HSV, shift H to get H^SV. H^SV to L^A^B^. L^A^B^ to LA^B^."""


    return


if __name__ == '__main__':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

    for i, data in enumerate(trainloader):
        images, _ = data
        image = Image.fromarray(images.astype('uint8'), 'RGB')
        image.save('trial.jpg')

        break




