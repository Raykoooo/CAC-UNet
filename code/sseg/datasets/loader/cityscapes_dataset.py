import numpy as np
from .dataset import BaseDataset
from ..aug.segaug import crop_aug
from ...models.registry import DATASET

@DATASET.register("CityscapesDataset")
class CityscapesDataset(BaseDataset):
    # overwrite
    def transform_mask(self, label):
        return label
    
    # overwrite
    def aug(self, image, label):
        return crop_aug(image, label, 512, 1024, scale=(0.44, 1), ratio=(2, 2))

@DATASET.register("origin-CityscapesDataset")
class CityscapesDataset(BaseDataset):
    # overwrite
    def transform_mask(self, label):
        return label
    
    # overwrite
    def aug(self, image, label):
        return crop_aug(image, label, 512, 1024, scale=(1.0, 1.0), ratio=(2, 2))


@DATASET.register("small-CityscapesDataset")
class SmallCityscapesDataset(BaseDataset):
    # overwrite
    def transform_mask(self, label):
        return label
    
    # overwrite
    def aug(self, image, label):
        return crop_aug(image, label, 256, 512, scale=(1, 1), ratio=(2, 2))


@DATASET.register("small-crop-CityscapesDataset")
class SmallCityscapesDataset(BaseDataset):
    # overwrite
    def transform_mask(self, label):
        return label
    
    # overwrite
    def aug(self, image, label):
        return crop_aug(image, label, 256, 512, scale=(1/4*(0.75)**2, 1/4*(1.25)**2), ratio=(2, 2))
