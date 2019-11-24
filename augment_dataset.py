import pickle
from PIL import Image, ImageCms
import numpy as np
from dataloader import CityScapesDataLoader
from torch.utils.data import DataLoader

import torch
import torchvision
import torchvision.transforms as transforms

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

def extract_pallete(img):
    return 

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
    h = (h+50)%255
    img_hsv[:,:,0] = h

    #retain original luminance
    img_rgb = Image.fromarray(img_hsv, mode="HSV").convert('RGB')
    img_lab_t = np.asarray(ImageCms.applyTransform(img_rgb, rgb2lab)).copy()
    img_lab_t[:,:,0] = L

    #convert LAB to RGB  
    img_recolor = Image.fromarray(img_lab_t, mode="LAB")
    lab2rgb = ImageCms.buildTransformFromOpenProfiles(lab_p, srgb_p, "LAB", "RGB")
    img_recolor = ImageCms.applyTransform(img_recolor, lab2rgb)
    
    img_recolor.save('recolor.jpg')

    return


if __name__ == '__main__':
    # transform = transforms.Compose(
    # [transforms.ToTensor()])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                     download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000,
    #                                       shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                    download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4,
    #                                      shuffle=False, num_workers=2)
    
    trainset = CityScapesDataLoader('/shared/medhini/data/cityscapes/')
    trainloader = DataLoader(trainset, batch_size=1, num_workers=0)

    for i,data in enumerate(trainloader, 0):
        image = np.asarray(data[0])
        image = Image.fromarray(np.uint8(image), 'RGB')

        # image_palette = extract_pallete(image)
        image_recolorized = hue_shift(image)
        break
    




