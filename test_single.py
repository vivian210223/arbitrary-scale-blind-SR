import argparse
from PIL import Image
import pdb

import utils

import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--hr', help='HR path', default='your path to estimate HR image.png')
	parser.add_argument('--sr', default='your path to estimate SR image.png')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
	# load file
	hr = transforms.ToTensor()(Image.open(args.hr).convert('RGB')).unsqueeze(0).to(device)
	sr = transforms.ToTensor()(Image.open(args.sr).convert('RGB')).unsqueeze(0).to(device)
	
	l = lpips(sr, hr)
	psnr = utils.calc_psnr(sr, hr)
	ssim = utils.calc_ssim(sr, hr)
	print('psnr: {:.4f}, ssim: {:.4f}, lpips: {:.4f}'.format(psnr, ssim, l))

