import torch
from torch import nn
import torch.nn.functional as F

import pdb

import models
from models import register

@register('simsiam')
class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, dim=256, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        #self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        self.encoder = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        # build a 3-layer projector
        prev_dim = self.encoder[-4].weight.shape[1]
        self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(prev_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
                                
        #self.projector.bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.out_dim = dim
    def forward(self, x1, x2=None):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        if x2 is not None:
            # compute features for one view
            f1 = self.encoder(x1).squeeze(-1).squeeze(-1) # NxC
            f2 = self.encoder(x2).squeeze(-1).squeeze(-1) # NxC
        
            z1 = self.projector(f1) # NxC
            z2 = self.projector(f2) # NxC
        
            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC
        
            return p1, p2, z1.detach(), z2.detach(), f1
        else:
            fea = self.encoder(x1).squeeze(-1).squeeze(-1) # NxC
            return fea


