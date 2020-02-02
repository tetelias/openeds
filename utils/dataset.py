from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


class EdsDS(Dataset):
    """Custom Dataset for loading eds dataset"""

    def __init__(self, fldr, images_dir='images/', mask_dir='labels/', mask=True, channels=1, transform=None):

        file_names = [f.stem for f in (Path(fldr)/images_dir).iterdir()]
        self.channels = channels
        self.img_paths = [Path(fldr)/images_dir/(f+'.png') for f in file_names]
        self.mask = mask
        if self.mask:
            self.mask_paths = [Path(fldr)/mask_dir/(f+'.npy') for f in file_names]
        self.transform = transform       

    def __getitem__(self, index):
        if self.channels==1:
            img = cv2.imread(str(self.img_paths[index]))[:,:,0][:,:,None]
        elif self.channels==3:
            img = cv2.imread(str(self.img_paths[index]))
        if self.mask:
            mask = np.load(self.mask_paths[index])
        else:
            mask = np.zeros(img.shape)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        return img, (mask*255).long().squeeze(0)

    def __len__(self):
        return len(self.img_paths)