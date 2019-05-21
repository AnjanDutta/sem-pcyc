#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system and other libraries
import os
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageOps


class DataGeneratorPaired(data.Dataset):
    def __init__(self, dataset, root, photo_dir, sketch_dir, photo_sd, sketch_sd, fls_sk, fls_im, clss,
                 transforms_sketch=None, transforms_image=None):
        self.dataset = dataset
        self.root = root
        self.photo_dir = photo_dir
        self.sketch_dir = sketch_dir
        self.photo_sd = photo_sd
        self.sketch_sd = sketch_sd
        self.fls_sk = fls_sk
        self.fls_im = fls_im
        self.clss = clss
        self.transforms_sketch = transforms_sketch
        self.transforms_image = transforms_image

    def __getitem__(self, item):
        sk = ImageOps.invert(Image.open(os.path.join(self.root, self.sketch_dir, self.sketch_sd, self.fls_sk[item]))).\
            convert(mode='RGB')
        im = Image.open(os.path.join(self.root, self.photo_dir, self.photo_sd, self.fls_im[item])).convert(mode='RGB')
        cls = self.clss[item]
        if self.transforms_image is not None:
            im = self.transforms_image(im)
        if self.transforms_sketch is not None:
            sk = self.transforms_sketch(sk)
        return sk, im, cls

    def __len__(self):
        return len(self.clss)

    def get_weights(self):
        weights = np.zeros(self.clss.shape[0])
        uniq_clss = np.unique(self.clss)
        for cls in uniq_clss:
            idx = np.where(self.clss == cls)[0]
            weights[idx] = 1 / idx.shape[0]
        return weights


class DataGeneratorSketch(data.Dataset):
    def __init__(self, dataset, root, sketch_dir, sketch_sd, fls_sk, clss_sk, transforms=None):
        self.dataset = dataset
        self.root = root
        self.sketch_dir = sketch_dir
        self.sketch_sd = sketch_sd
        self.fls_sk = fls_sk
        self.clss_sk = clss_sk
        self.transforms = transforms

    def __getitem__(self, item):
        sk = ImageOps.invert(Image.open(os.path.join(self.root, self.sketch_dir, self.sketch_sd, self.fls_sk[item]))).\
            convert(mode='RGB')
        cls_sk = self.clss_sk[item]
        if self.transforms is not None:
            sk = self.transforms(sk)
        return sk, cls_sk

    def __len__(self):
        return len(self.fls_sk)


class DataGeneratorImage(data.Dataset):
    def __init__(self, dataset, root, photo_dir, photo_sd, fls_im, clss_im, transforms=None):

        self.dataset = dataset
        self.root = root
        self.photo_dir = photo_dir
        self.photo_sd = photo_sd
        self.fls_im = fls_im
        self.clss_im = clss_im
        self.transforms = transforms

    def __getitem__(self, item):
        im = Image.open(os.path.join(self.root, self.photo_dir, self.photo_sd, self.fls_im[item])).convert(mode='RGB')
        cls_im = self.clss_im[item]
        if self.transforms is not None:
            im = self.transforms(im)
        return im, cls_im

    def __len__(self):
        return len(self.fls_im)
