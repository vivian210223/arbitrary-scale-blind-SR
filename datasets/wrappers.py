import functools
import random
import math
from PIL import Image
import pdb
import numpy as np

import torch
from torch.utils.data import Dataset

from datasets import register
from utils import to_pixel_samples, make_coord

# with pair data
@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] / img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_hr, w_hr = img_hr.shape[-2:] 
            h_lr, w_lr = img_lr.shape[-2:]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
        shape = torch.tensor(crop_hr.shape)
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]
        
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'shape': shape
        }

# for degradation model
@register('sr-degrade')
class ScaleDownsampled_degrade(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, scale=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.scale = scale
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, s = idx
        if self.scale:
            s = self.scale
        img = self.dataset[idx]
        
        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            crop_hr = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            
            
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            try:
                x0 = random.randint(0, img.shape[-2] - w_hr)
                y0 = random.randint(0, img.shape[-1] - w_hr)
                x1 = random.randint(0, img.shape[-2] - w_hr)
                y1 = random.randint(0, img.shape[-1] - w_hr)
            except:
                pdb.set_trace()
            # clean
            crop_q = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_k = img[:, x1: x1 + w_hr, y1: y1 + w_hr]
                
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5
            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_k = augment(crop_k)
            crop_q = augment(crop_q)

        
        return {
            'query': crop_q,
            'key': crop_k,
            'scale': s
        }


# for SR model/ gaussian blur
@register('sr-gaussian')
class ScaleDownsampled_gaussian(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None, scale=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q
        self.scale = scale

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx, s = idx
        if self.scale:
            s = self.scale
        img = self.dataset[idx]
                
        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            crop_img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            
            # clean
            crop_img = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
                
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5
            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_img = augment(crop_img)

        hr_coord, hr_rgb = to_pixel_samples(crop_img.contiguous())
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
            
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_img.shape[-2]
        cell[:, 1] *= 2 / crop_img.shape[-1]   
        
        return {
            'gt': hr_rgb,
            'cell': cell,
            'coord': hr_coord,
            'scale': s,
            'inp': crop_img
        }

