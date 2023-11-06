import os
import time
import math
import numpy as np
import pdb
import cv2

import torch
from torch.optim import SGD, Adam, AdamW
import torch.nn as nn
import torch.nn.functional as F


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img, patch=False):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    if patch:
        coord = make_coord(img.shape[-2:], flatten=False)
        rgb = img.permute(1, 2, 0)
    else:
        coord = make_coord(img.shape[-2:], flatten=True)
        rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)

def ssim(img1, img2):
    C1 = (0.01)**2
    C2 = (0.03)**2
    img1 = img1.cpu().numpy()
    img2 = img2.cpu().numpy()
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if img1.size(1) > 1:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = img1.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        img1 = img1.mul(convert).sum(dim=1)
        img2 = img2.mul(convert).sum(dim=1)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def get_model(DDP, model):
    if not DDP:
        return model
    else:
        return model.module

def freeze_unfreeze(model, model_state):
    if model_state == 'freeze':
        for p in model.parameters():
            p.requires_grad = False
    elif model_state == 'unfreeze':
        for p in model.parameters():
            p.requires_grad = True
    else:
        pass
    return model

def save(model, optimizer, opspec, epoch, DDP, savepath, state):
    if isinstance(model, dict):
        model_spec = {}
        for k, v in model.items():
            m = get_model(DDP, v)
            model_spec[k] = m.state_dict()
            keys = list(model_spec[k].keys())
            for key in keys:
                if 'encoder_k' in key or 'queue' in key:
                    del model_spec[k][key]
    else:    
        m = get_model(DDP, model)
        model_spec = m.state_dict()
        keys = list(model_spec.keys())
        for key in keys:
            if 'encoder_k' in key or 'queue' in key:
                del model_spec[key]
    # optimizer
    optimizer_spec = opspec
    optimizer_spec['sd'] = optimizer.state_dict()
    
    sv_file = {
        'model': model_spec,
        'optimizer': optimizer_spec,
        'epoch': epoch
    }
    if state=='last':
        torch.save(sv_file, os.path.join(savepath, 'epoch-last.pth'))
    elif state=='hun':
        torch.save(sv_file,
            os.path.join(savepath, 'epoch-{}.pth'.format(epoch)))
    elif state=='best':
        torch.save(sv_file, os.path.join(savepath, 'epoch-best.pth'))
    
        
def wave(inp, xfm):
    if inp.shape[-1]==3:
        tem = int(math.sqrt(inp.shape[-2]))
        inp = inp.permute(0, 2, 1).reshape(-1, 3, tem, tem)
    # wavelet transform, to encoder
    _, Yh = xfm(inp.view(-1, 3, inp.shape[-2], inp.shape[-1]))
    HL, LH, HH = torch.unbind(Yh[0], dim=2)
    HH = HH.reshape(-1, 3, HH.shape[-2], HH.shape[-1])
    LH = LH.reshape(-1, 3, LH.shape[-2], LH.shape[-1])
    HL = HL.reshape(-1, 3, HL.shape[-2], HL.shape[-1])
    H = torch.cat((HH, LH, HL), dim=1)
    return H

def Kernel(lr_blur, lr_clean, ksize, eps=1e-20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    shape = lr_blur.shape
    blur_ft = torch.fft.fft2(lr_blur, dim=(-3, -2, -1))
    blur_ft = torch.stack((blur_ft.real, blur_ft.imag), -1)
    clean_ft = torch.fft.fft2(lr_clean, dim=(-3, -2, -1))
    clean_ft = torch.stack((clean_ft.real, clean_ft.imag), -1)

    denominator = clean_ft[:, :, :, :, 0] * clean_ft[:, :, :, :, 0] \
                  + clean_ft[:, :, :, :, 1] * clean_ft[:, :, :, :, 1] + eps

    inv_denominator = torch.zeros_like(clean_ft)
    inv_denominator[:, :, :, :, 0] = clean_ft[:, :, :, :, 0] / denominator
    inv_denominator[:, :, :, :, 1] = -clean_ft[:, :, :, :, 1] / denominator

    kernel = torch.zeros_like(blur_ft).to(device)
    kernel[:, :, :, :, 0] = inv_denominator[:, :, :, :, 0] * blur_ft[:, :, :, :, 0] \
                            - inv_denominator[:, :, :, :, 1] * blur_ft[:, :, :, :, 1]
    kernel[:, :, :, :, 1] = inv_denominator[:, :, :, :, 0] * blur_ft[:, :, :, :, 1] \
                            + inv_denominator[:, :, :, :, 1] * blur_ft[:, :, :, :, 0]
    psf = torch.fft.ifft2(torch.complex(kernel[..., 0], kernel[..., 1]), dim=(-3, -2, -1))
    # circularly shift
    centre = ksize[-1]//2 + 1
    ker = torch.zeros(ksize).to(device)
    
    ker[:, :, (centre-1):, (centre-1):] = psf[:, :, :centre, :centre]#.mean(dim=1, keepdim=True)
    ker[:, :, (centre-1):, :(centre-1)] = psf[:, :, :centre, -(centre-1):]#.mean(dim=1, keepdim=True)
    ker[:, :, :(centre-1), (centre-1):] = psf[:, :, -(centre-1):, :centre]#.mean(dim=1, keepdim=True)
    ker[:, :, :(centre-1), :(centre-1)] = psf[:, :, -(centre-1):, -(centre-1):]#.mean(dim=1, keepdim=True)
    return ker

def gt_lr_kernel(lr_blur, lr_clean, ksize=21, eps=1e-20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ks = []
    mask = torch.ones((lr_blur.shape[0], 1, ksize, ksize)).to(device)
    for c in range(lr_blur.shape[1]):
        k_correct = Kernel(lr_blur[:, c:c+1, ...], lr_clean[:, c:c+1, ...], mask.size(), eps)
        ks.append(k_correct.clone())
        mask *= k_correct
    ks = torch.cat(ks, dim=1)
    k_correct = torch.mean(ks, dim=1, keepdim=True) * (mask>0)
    k_correct = zeroize_negligible_val(k_correct, n=round(0.05*ksize**2))
    
    if ksize < 21:
        pad = (21-ksize)//2
        k_correct = F.pad(k_correct, (pad, pad, pad, pad) , "constant", 0)
    
    return k_correct
    
def zeroize_negligible_val(k, n=40):
    """Zeroize values that are negligible w.r.t to values in k"""
    # Sort K's values in order to find the n-th largest
    pc = k.shape[-1]//2 + 1
    k_sorted, indices = torch.sort(k.flatten(start_dim=1))
    # Define the minimum value as the 0.75 * the n-th largest value
    k_n_min = 0.75 * k_sorted[:, -n - 1]
    # Clip values lower than the minimum value
    filtered_k = torch.clamp(k - k_n_min.view(-1, 1, 1, 1), min=0, max=1.0)
    filtered_k[:, :, pc, pc] += 1e-20
    # Normalize to sum to 1
    norm_k = filtered_k / torch.sum(filtered_k, dim=(2, 3), keepdim=True)
    return norm_k