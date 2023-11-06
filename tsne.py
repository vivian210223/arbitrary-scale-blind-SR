import glob
import imageio
import numpy as np
from sklearn.manifold import TSNE
import pdb
from pytorch_wavelets import DWTForward
import yaml
from tqdm import tqdm

import torch

import models
import utils
from datasets.blur import SRMDPreprocessing

if __name__ == '__main__':
	#### degradation settings
	blur_kernel = [21, 21, 21, 21, 21, 13]
	blur_type = 'aniso_gaussian'
	scale = [2.5, 1.7, 2.8, 3.2, 2 ,3.5]
	noise = 0

	lambda_1_list = [0.5, 1.1, 3.5,  3.2, 2.1, 0]
	lambda_2_list = [0.5, 5.5, 3.5,  1.5, 4.0, 0]
	theta_list    = [0, 125, 0,  125, 120, 45]
	
	# paths
	img_path = 'your path to test HR datasets directory/*.png'
	model_path = 'your path to model weight.pth'
	tsne_path = 'your path to save tsne.png'
	config_path = 'your path to model config.yaml'

	#### read config file
	with open(config_path, 'r') as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = models.make(config['degrade'], load_sd=model_path, freeze=True, key='degrade').to(device)
	kernel = models.make(config['kernel'], load_sd=model_path, freeze=True, key='kernel').to(device) 

	model.eval()
	
	HR_img_list = glob.glob(img_path)
	fea_list, label = [], []
	wav = DWTForward(**config['wavelet']).to(device)
	
	for s, lambda_1, lambda_2, theta, ks in zip(scale, lambda_1_list, lambda_2_list, theta_list, blur_kernel):
		degrade = SRMDPreprocessing(
						kernel_size=ks,
						blur_type=blur_type,
						lambda_1=lambda_1,
						lambda_2=lambda_2,
						theta=theta,
						noise=noise)

		with torch.no_grad():
			for i in tqdm(range(len(HR_img_list)), leave=False, desc='tsne'):
				# read HR images
				HR_img = imageio.imread(HR_img_list[i])
				if np.ndim(HR_img) < 3: #gray img
					HR_img = np.stack([HR_img, HR_img, HR_img], 2)
				HR_img = np.ascontiguousarray(HR_img.transpose((2, 0, 1)))
				HR_img = torch.from_numpy(HR_img).float().to(device).unsqueeze(0)
				b, c, h, w = HR_img.size()
				
				# generate LR images
				LR_img, ks = degrade(HR_img, scale=s, state='test+kernel', random=False, norm=False)
				LR_img, LR_clean = torch.split(LR_img, 1)
				H = utils.wave(LR_img, wav)
				fea = model(x1=H)
				fea_list.append(fea.cpu().numpy())

	f = np.concatenate(fea_list, 0)

	f_min = np.min(f)
	f_max = np.max(f)
	# normalization
	f_norm = (f - f_min) / (f_max - f_min)
	# T-SNE
	tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=30, early_exaggeration=12, learning_rate=50)
	embed = tsne.fit_transform(f_norm)
	embed = embed.reshape(len(lambda_1_list), 1, 100, -1)
	
	# visualization
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(5, 5))
	ax = plt.subplot(111)
	p1 = ax.scatter(embed[0, 0, :, 0], embed[0, 0, :, 1], c='b')
	p2 = ax.scatter(embed[1, 0, :, 0], embed[1, 0, :, 1], c='r')
	p3 = ax.scatter(embed[2, 0, :, 0], embed[2, 0, :, 1], c='g')
	p4 = ax.scatter(embed[3, 0, :, 0], embed[3, 0, :, 1], c='k')
	p5 = ax.scatter(embed[4, 0, :, 0], embed[4, 0, :, 1], c='m')
	p6 = ax.scatter(embed[5, 0, :, 0], embed[5, 0, :, 1], c='c')
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.savefig(tsne_path)
	