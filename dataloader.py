import os
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

if __name__=='__main__':
    trainset = CityScapesDataLoader('/shared/medhini/data/cityscapes/')
    trainloader = DataLoader(trainset, batch_size=1, num_workers=0)

    for i,data in enumerate(trainloader, 0):
        image = np.asarray(data[0])
        image = Image.fromarray(np.uint8(image), 'RGB')
        image.save('trial.jpg')
        break



