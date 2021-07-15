import os
from os.path import join as pjoin
import collections
import torch
import numpy as np
from glob import glob
from random import shuffle
import re

from PIL import Image
from torch.utils import data
from torchvision import transforms
from augmentor.Pipeline import Pipeline
from utils import gray2rgb, gray2rgbTorch

class ImageFolderLoader(data.Dataset):

    def __init__(
        self,
        root,
        sample_weight=None,
        is_transform=True,
        img_size=256,
        split="train",
        test_mode=False,
        img_norm=True,
        n_classes=8,

    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.is_augmentations = True
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.mean = np.array([125.08347, 124.99436, 124.99769])
        self.files = collections.defaultdict(list)
        self.img_size = img_size
        self.files = collections.defaultdict(list)
        self.n_classes = n_classes
        self.sample_weight = sample_weight

        for split in ["train", "test"]:
            file_list = []
            for sub_classes in os.listdir(self.root + split):
                path = pjoin(root, split, sub_classes + "/*png")
                file_list += glob(path)
            self.files[split] = file_list
            # self.setup_annotations()

        # normMean = [0.498, 0.497, 0.497]
        # normStd = [0.206, 0.206, 0.206]
        normMean = [0.498]
        normStd = [0.206]

        self.tf = transforms.Compose(
            [
                #transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                #transforms.Normalize(normMean, normStd),
            ]
        )

    def __len__(self):
        k = len(self.files[self.split])
        return k

    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        im_name_split = re.split(r'[/\\]', im_name[:-4])
        im = Image.open(im_name)
        lbl = int(re.split(r'[/\\]', im_name)[-2])
        img_name = im_name_split[-3] + '/' + im_name_split[-2] + '/' + im_name_split[-1]
        if self.is_transform:
            im = self.transform(im)
        sample_weight_bs = self.sample_weight[index]
        return im, lbl ,img_name, sample_weight_bs



    def transform(self, img):
        if img.size == self.img_size:
            pass
        else:
            img = img.resize(self.img_size, Image.ANTIALIAS)  # uint8 with RGB mode
        img_rgb = self.tf(img)
        return img_rgb


    def augmentations(self, img, lbl = None):
        if lbl is None:
            lbl = img
        p = Pipeline(img, lbl)
        # Add operations to the pipeline as normal:
        p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
        p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
        p.zoom_random(probability=0.5, percentage_area=0.8)
        p.skew_left_right(probability=0.5)
        p.flip_left_right(probability=0.5)
        img2, lbl2 = p.sample()
        return img2
