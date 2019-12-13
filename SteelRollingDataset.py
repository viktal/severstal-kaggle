import os
import cv2
import numpy as np
import pandas as pd
from utils import rle2mask
from torch.utils.data import Dataset
from albumentations import Compose
from typing import List, Dict


class SteelRollingDataset(Dataset):
    def __init__(self, n_classes, files: List[str], rles=None, transforms: Compose = None,
                 preload=False, rle_height: int = 256, rle_witdh: int = 1600):
        super().__init__()
        assert rles is None or len(files) == len(rles)
        self.files = files
        self.preload = preload
        if self.preload:
            self.images = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in self.files]
        self.rles = rles if rles else None
        self.classes = n_classes
        self.transforms = transforms
        self.rle_height = rle_height
        self.rle_width = rle_witdh

    def _to_per_class_mask(self, rles):
        one_hot_mask = np.zeros((self.rle_height, self.rle_width, self.classes), dtype=np.uint8)
        for cls, rle in rles.items():
            one_hot_mask[:, :, cls] = rle2mask(rle, self.rle_height, self.rle_width)
        # background == not occupied by any clas
        one_hot_mask[:, :, 0] = 1 - one_hot_mask.max(axis=-1, keepdims=False, initial=0)
        return one_hot_mask

    def filename(self, index):
        return self.files[index]

    def image(self, index):
        if self.preload:
            return self.images[index]
        return cv2.imread(self.filename(index), cv2.IMREAD_UNCHANGED)

    def __getitem__(self, index):
        img = self.image(index)
        to_CHW = lambda ndarray: np.transpose(ndarray, [2, 0, 1])
        if self.rles:
            mask = self._to_per_class_mask(self.rles[index])
            augmented = self.transforms(image=img, mask=mask)
            ret = to_CHW(augmented['image']), to_CHW(augmented['mask'])
        else:
            augmented = self.transforms(image=img)
            ret = to_CHW(augmented['image']), self.filename(index)
        return ret

    def __len__(self):
        return len(self.files)
