import math
import numpy as np
import pdb
import random as ran

import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_sigma(sig_x, sig_y, radians):
    sig_x = sig_x.view(-1, 1, 1)
    sig_y = sig_y.view(-1, 1, 1)
    radians = radians.view(-1, 1, 1)

    D = torch.cat([F.pad(sig_x ** 2, [0, 1, 0, 0]), F.pad(sig_y ** 2, [1, 0, 0, 0])], 1)
    U = torch.cat([torch.cat([radians.cos(), -radians.sin()], 2),
                   torch.cat([radians.sin(), radians.cos()], 2)], 1)
    sigma = torch.bmm(U, torch.bmm(D, U.transpose(1, 2)))

    return sigma


def anisotropic_gaussian_kernel(device, batch, kernel_size, covar):
    ax = torch.arange(kernel_size).float().to(device) - kernel_size // 2

    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    xy = torch.stack([xx, yy], -1).view(batch, -1, 2)

    inverse_sigma = torch.inverse(covar)
    kernel = torch.exp(- 0.5 * (torch.bmm(xy, inverse_sigma) * xy).sum(2)).view(batch, kernel_size, kernel_size)

    return kernel / kernel.sum([1, 2], keepdim=True)


def isotropic_gaussian_kernel(device, batch, kernel_size, sigma):
    ax = torch.arange(kernel_size).float().to(device) - kernel_size//2
    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2. * sigma.view(-1, 1, 1) ** 2))

    return kernel / kernel.sum([1,2], keepdim=True)


def random_anisotropic_gaussian_kernel(device, batch=1, kernel_size=21, lambda_min=0.2, lambda_max=4.0):
    theta = torch.rand(batch).to(device) * math.pi
    lambda_1 = torch.rand(batch).to(device) * (lambda_max - lambda_min) + lambda_min
    lambda_2 = torch.rand(batch).to(device) * (lambda_max - lambda_min) + lambda_min

    covar = cal_sigma(lambda_1, lambda_2, theta)
    kernel = anisotropic_gaussian_kernel(device, batch, kernel_size, covar)
    return kernel


def stable_anisotropic_gaussian_kernel(device, kernel_size=21, theta=0, lambda_1=0.2, lambda_2=4.0):
    theta = torch.ones(1).to(device) * theta / 180 * math.pi
    lambda_1 = torch.ones(1).to(device) * lambda_1
    lambda_2 = torch.ones(1).to(device) * lambda_2

    covar = cal_sigma(lambda_1, lambda_2, theta)
    kernel = anisotropic_gaussian_kernel(device, 1, kernel_size, covar)
    return kernel


def random_isotropic_gaussian_kernel(device, batch=1, kernel_size=21, sig_min=0.2, sig_max=4.0):
    x = torch.rand(batch).to(device) * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel(device, batch, kernel_size, x)
    return k


def stable_isotropic_gaussian_kernel(device, kernel_size=21, sig=4.0):
    x = torch.ones(1).to(device) * sig
    k = isotropic_gaussian_kernel(device, 1, kernel_size, x)
    return k


def random_gaussian_kernel(batch, device, kernel_size=21, blur_type='iso_gaussian', sig_min=0.2, sig_max=4.0, lambda_min=0.2, lambda_max=4.0):
    if blur_type == 'iso_gaussian':
        return random_isotropic_gaussian_kernel(device=device, batch=batch, kernel_size=kernel_size, sig_min=sig_min, sig_max=sig_max)
    elif blur_type == 'aniso_gaussian':
        return random_anisotropic_gaussian_kernel(device=device, batch=batch, kernel_size=kernel_size, lambda_min=lambda_min, lambda_max=lambda_max)


def stable_gaussian_kernel(device, kernel_size=21, blur_type='iso_gaussian', sig=2.6, lambda_1=0.2, lambda_2=4.0, theta=0):
    if blur_type == 'iso_gaussian':
        return stable_isotropic_gaussian_kernel(device=device, kernel_size=kernel_size, sig=sig)
    elif blur_type == 'aniso_gaussian':
        return stable_anisotropic_gaussian_kernel(device=device, kernel_size=kernel_size, lambda_1=lambda_1, lambda_2=lambda_2, theta=theta)


# implementation of matlab bicubic interpolation in pytorch
class bicubic(nn.Module):
    def __init__(self, device):
        super(bicubic, self).__init__()
        self.device = device
    def cubic(self, x):
        absx = torch.abs(x)
        absx2 = torch.abs(x) * torch.abs(x)
        absx3 = torch.abs(x) * torch.abs(x) * torch.abs(x)

        condition1 = (absx <= 1).to(torch.float32)
        condition2 = ((1 < absx) & (absx <= 2)).to(torch.float32)

        f = (1.5 * absx3 - 2.5 * absx2 + 1) * condition1 + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * condition2
        return f

    def contribute(self, in_size, out_size, scale):
        kernel_width = 4
        if scale < 1:
            kernel_width = 4 / scale # float
        x0 = torch.arange(start=1, end=out_size[0] + 1).to(torch.float32).to(self.device) # 48
        x1 = torch.arange(start=1, end=out_size[1] + 1).to(torch.float32).to(self.device) # 48

        # upsample
        u0 = x0 / scale + 0.5 * (1 - 1 / scale) #48
        u1 = x1 / scale + 0.5 * (1 - 1 / scale) #48

        left0 = torch.floor(u0 - kernel_width / 2) #48
        left1 = torch.floor(u1 - kernel_width / 2) #48

        P = np.ceil(kernel_width) + 2 # float, 11

        indice0 = left0.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).to(self.device) #48,11
        indice1 = left1.unsqueeze(1) + torch.arange(start=0, end=P).to(torch.float32).unsqueeze(0).to(self.device)
        mid0 = u0.unsqueeze(1) - indice0.unsqueeze(0) #1, 48, 11
        mid1 = u1.unsqueeze(1) - indice1.unsqueeze(0)

        if scale < 1:
            weight0 = scale * self.cubic(mid0 * scale) #1, 48, 11
            weight1 = scale * self.cubic(mid1 * scale)
        else:
            weight0 = self.cubic(mid0)
            weight1 = self.cubic(mid1)

        #nornalize
        weight0 = weight0 / (torch.sum(weight0, 2).unsqueeze(2)) # 1, 48, 11
        weight1 = weight1 / (torch.sum(weight1, 2).unsqueeze(2))
        # smaller than indice be 1, 1 48, 11
        indice0 = torch.min(torch.max(torch.FloatTensor([1]).to(self.device), indice0), torch.FloatTensor([in_size[0]]).to(self.device)).unsqueeze(0)
        indice1 = torch.min(torch.max(torch.FloatTensor([1]).to(self.device), indice1), torch.FloatTensor([in_size[1]]).to(self.device)).unsqueeze(0)

        kill0 = torch.eq(weight0, 0)[0][0] #11
        kill1 = torch.eq(weight1, 0)[0][0] #11

        weight0 = weight0[:, :, kill0 == 0] #1,48,8
        weight1 = weight1[:, :, kill1 == 0]

        indice0 = indice0[:, :, kill0 == 0] #1, 48, 8
        indice1 = indice1[:, :, kill1 == 0]

        return weight0, weight1, indice0, indice1

    def forward(self, input, scale=1/4):
        b, c, h, w = input.shape
        weight0, weight1, indice0, indice1 = self.contribute([h, w], [round(h * scale), round(w * scale)], scale)
        weight0 = weight0[0] #48, 8
        weight1 = weight1[0]

        indice0 = indice0[0].long()
        indice1 = indice1[0].long()

        out = input[:, :, (indice0 - 1), :] * (weight0.unsqueeze(0).unsqueeze(1).unsqueeze(4)) #16,3,48,8,96
        out = (torch.sum(out, dim=3)) # 16, 3, 48, 96
        A = out.permute(0, 1, 3, 2)

        out = A[:, :, (indice1 - 1), :] * (weight1.unsqueeze(0).unsqueeze(1).unsqueeze(4))
        out = out.sum(3).permute(0, 1, 3, 2) #16, 3, 48, 48

        return out


class Gaussin_Kernel(object):
    def __init__(self, device, blur_type='iso_gaussian',
                 sig=2.6, sig_min=0.2, sig_max=4.0,
                 lambda_1=0.2, lambda_2=4.0, theta=0, lambda_min=0.2, lambda_max=4.0):
        self.blur_type = blur_type

        self.sig = sig
        self.sig_min = sig_min
        self.sig_max = sig_max

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.theta = theta
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        self.device = device

    def __call__(self, batch, random, kernel_size):
        # random kernel
        if random == True:
            return random_gaussian_kernel(batch, device=self.device, kernel_size=kernel_size, blur_type=self.blur_type,
                                          sig_min=self.sig_min, sig_max=self.sig_max,
                                          lambda_min=self.lambda_min, lambda_max=self.lambda_max)

        # stable kernel
        else:
            return stable_gaussian_kernel(device=self.device, kernel_size=kernel_size, blur_type=self.blur_type,
                                          sig=self.sig, lambda_1=self.lambda_1, lambda_2=self.lambda_2, theta=self.theta)

class BatchBlur(nn.Module):
    def __init__(self):
        super(BatchBlur, self).__init__()
    def forward(self, input, kernel, kernel_size):
        if kernel_size % 2 == 1:
            self.pad = nn.ReflectionPad2d(kernel_size//2)
        else:
            self.pad = nn.ReflectionPad2d((kernel_size//2, kernel_size//2-1, kernel_size//2, kernel_size//2-1))
        
        B, C, H, W = input.size()
        
        input_pad = self.pad(input)
        H_p, W_p = input_pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = input_pad.view((C * B, 1, H_p, W_p))
            kernel = kernel.contiguous().view((1, 1, kernel_size, kernel_size))

            return F.conv2d(input_CBHW, kernel, padding=0).view((B, C, H, W))
        else:
            input_CBHW = input_pad.view((1, C * B, H_p, W_p))
            kernel = kernel.contiguous().view((B, 1, kernel_size, kernel_size))
            kernel = kernel.repeat(1, C, 1, 1).view((B * C, 1, kernel_size, kernel_size))

            return F.conv2d(input_CBHW, kernel, groups=B*C).view((B, C, H, W))



class SRMDPreprocessing(object):
    def __init__(self, 
                 mode='bicubic',
                 kernel_size=[21],#[7,9,11,13,15,17,19,21],
                 blur_type='aniso_gaussian',
                 sig=2.6,
                 sig_min=0.2,
                 sig_max=4.0,
                 lambda_1=0.2,
                 lambda_2=4.0,
                 theta=0,
                 lambda_min=0.6,
                 lambda_max=5.0,#4.0,
                 noise=0.0,
                 ):
        '''
        # sig, sig_min and sig_max are used for isotropic Gaussian blurs
        During training phase (random=True):
            the width of the blur kernel is randomly selected from [sig_min, sig_max]
        During test phase (random=False):
            the width of the blur kernel is set to sig

        # lambda_1, lambda_2, theta, lambda_min and lambda_max are used for anisotropic Gaussian blurs
        During training phase (random=True):
            the eigenvalues of the covariance is randomly selected from [lambda_min, lambda_max]
            the angle value is randomly selected from [0, pi]
        During test phase (random=False):
            the eigenvalues of the covariance are set to lambda_1 and lambda_2
            the angle value is set to theta
        '''
        self.kernel_size = kernel_size
        self.theta = theta
        self.mode = mode
        self.noise = noise
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gen_kernel = Gaussin_Kernel(device=self.device,
             blur_type=blur_type,
            sig=sig, sig_min=sig_min, sig_max=sig_max,
            lambda_1=lambda_1, lambda_2=lambda_2, theta=theta,
             lambda_min=lambda_min, lambda_max=lambda_max
        )
        self.blur = BatchBlur()
        self.bicubic = bicubic(device=self.device)

    def __call__(self, img, scale, random, state, norm=True):
        with torch.no_grad():
            # w/ different img I/O, value different. default works on 0~255
            if norm:
                hr_tensor = img*255
            else:
                hr_tensor = img    
            dim5 = False
            if hr_tensor.dim() == 5:
                dim5 = True 
                B, N, C, H, W = hr_tensor.size()
            else:
                B, C, H, W = hr_tensor.size()
                
            if random:
                ks = ran.choice(self.kernel_size)
            else:
                ks = self.kernel_size
               
            # not blur
            if state == 'down' or (self.gen_kernel.blur_type == 'iso_gaussian' and self.gen_kernel.sig == 0):
                hr_blured = hr_tensor.view(-1, C, H, W)
                b_kernels = None
            # gaussian blur
            elif 'motion' in state:
                # blur
                b_kernels = motion_psf(ks, self.theta, B, random, self.device)
                hr_blured = self.blur(hr_tensor.view(B, -1, H, W), b_kernels, ks)
                hr_blured = hr_blured.view(-1, C, H, W)  # B, C, H, W
            else:
                b_kernels= self.gen_kernel(B, random, ks)  # B degradations
                # blur
                hr_blured = self.blur(hr_tensor.view(B, -1, H, W), b_kernels, ks)
                hr_blured = hr_blured.view(-1, C, H, W)  # B, C, H, W
            if 'kernel' in state:
                hr_tensor = hr_tensor.view(-1, C, H, W)
                if dim5:
                    N *= 2
                    hr_blured = torch.cat((hr_blured, hr_tensor), dim=1).view(-1, C, H, W)
                else:
                    B *= 2
                    hr_blured = torch.cat((hr_blured, hr_tensor), dim=0).view(-1, C, H, W)
                
            # downsampling
            if random:
                mode_list = ['bicubic', 'bilinear', 'area']
                mode = ran.choice(mode_list)
                if mode == 'bicubic':
                    lr_blured = self.bicubic(hr_blured, scale=1/scale)
                elif state=='gd':
                    lr_blured = F.interpolate(hr_blured, scale_factor=1/scale, mode=self.mode)
                else:
                    lr_blured = F.interpolate(hr_blured, size=[48, 48], mode=mode)
                    assert lr_blured.shape[-1] == 48
            elif self.mode == 'bicubic':
                lr_blured = self.bicubic(hr_blured, scale=1/scale)
            else: # bilinear, nearest
                lr_blured = F.interpolate(hr_blured, scale_factor=1/scale, mode=self.mode)
            
            # add noise
            if self.noise > 0:
                _, C, H_lr, W_lr = lr_blured.size()
                noise_level = torch.rand(B, 1, 1, 1).to(lr_blured.device) * self.noise #if random else self.noise
                noise = torch.randn_like(lr_blured).view(-1, C, H_lr, W_lr).mul_(noise_level).view(-1, C, H_lr, W_lr)
                lr_blured.add_(noise)

            if dim5:
                lr_blured = (torch.clamp(lr_blured.round(), 0, 255)/255)\
                            .view(B, N, C, lr_blured.shape[-2], lr_blured.shape[-1])
            else:
                lr_blured = (torch.clamp(lr_blured.round(), 0, 255)/255)\
                            .view(B, C, lr_blured.shape[-2], lr_blured.shape[-1])
            
            if 'kernel' in state:
                return lr_blured, ks
            return lr_blured
            