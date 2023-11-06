import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F

import models
from models import register
import utils

def weights_init(m):
	for layer in m.modules():
		if isinstance(layer, nn.Linear):
			nn.init.xavier_uniform_(layer.weight)

@register('kernel')
class KERNEL(nn.Module):
	def __init__(
                self, kernel_size=21, in_dim=256, filter_structures=[11, 7, 5, 1]):
		super(KERNEL, self).__init__()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.filter_structures = filter_structures
		self.ksize = kernel_size
		self.dec = nn.ModuleList()
		self.decoder = nn.Sequential(nn.Linear(in_dim, in_dim//4), nn.ReLU(),
										nn.Linear(in_dim//4, in_dim//16))
		for i, f_size in enumerate(self.filter_structures):
			self.dec.append(nn.Linear(in_dim//16, f_size**2))
		weights_init(self.dec)
		
	def calc_curr_k(self, kernels, batch):
		"""given a generator network, the function calculates the kernel it is imitating"""
		delta = torch.ones([1, batch]).unsqueeze(-1).unsqueeze(-1).to(self.device)
		for ind, w in enumerate(kernels):
			curr_k = F.conv2d(delta, w, padding=self.ksize - 1, groups=batch) if ind == 0 else F.conv2d(curr_k, w, groups=batch)
		curr_k = curr_k.reshape(batch, 1, self.ksize, self.ksize).flip([2, 3])
		return curr_k

	def forward(self, degrade):
		batch = degrade.shape[0]
		degrade = self.decoder(degrade)
		kernels = []	
		for i in range(len(self.filter_structures)):
			kernels.append(self.dec[i](degrade).reshape(
												batch, 1,
												self.filter_structures[i],
												self.filter_structures[i]))
		K = self.calc_curr_k(kernels, batch)
		K = K / torch.sum(K, dim=(2, 3), keepdim=True)

		return K

