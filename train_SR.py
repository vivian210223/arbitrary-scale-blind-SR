import argparse
import os
import pdb
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import datasets
from datasets.blur import SRMDPreprocessing
import models
import utils
from test import batched_predict
from models.controller import *
from datasets.queue import dequeue_and_enqueue


class Trainer():
	def __init__(self, config, args):
		#### distributed data parallel    
		if args.DDP:
			dist.init_process_group(backend='nccl')
			dist.barrier()
			self.local_rank = dist.get_rank()
			torch.cuda.set_device(self.local_rank)
			self.device = torch.device('cuda', self.local_rank)
			ngpu = dist.get_world_size()
		else:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			self.local_rank = 0 # just for dataloader
			ngpu = 1

		pool_type = 'SR'
		#### dataloader
		self.train_loader1, self.train_sampler1, self.val_loader1, self.val_sampler1,\
		 = datasets.make_data_loaders(config, args.DDP, self.local_rank, ngpu, state='SR')
		self.pool = dequeue_and_enqueue(config, pool_type)
		self.pool_val = dequeue_and_enqueue(config, pool_type)
		#### prepair training
		# model/optimzer/lr sceduler
		self.model = models.make(config['model'], args={'config': config}).to(self.device)
		
		if not args.DDP or dist.get_rank() == 0:
			print('model: #params={}'.format(utils.compute_num_params(self.model, text=True)))
			
		if args.DDP:
			self.model = DDP(self.model, device_ids=[self.local_rank],\
				 output_device=self.local_rank, find_unused_parameters=True)
		
		self.criterion = nn.L1Loss()
		self.optimizer = utils.make_optimizer(self.model.parameters(), config['optimizer'])
		# infer learning rate before changing batch size
		if config.get('multi_step_lr') is None:
			self.lr_scheduler = None
		else:
			self.lr_scheduler = MultiStepLR(self.optimizer, **config['multi_step_lr'])
		
		# else
		self.args = args
		self.config = config
		self.epoch_save = config.get('epoch_save')
		self.epoch_val = config.get('epoch_val')
		self.max_val_v = -1e18
		self.bs = config['total_batch_size']
		self.degrade = SRMDPreprocessing()

		# data normalize
		data_norm = self.config['data_norm']
		t = data_norm['inp']
		self.inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(self.device) 
		self.inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(self.device)
		t = data_norm['gt']
		self.gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(self.device)
		self.gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(self.device)


	def preprocess(self, batch, state):
		scale = batch['scale'][0].item()
		lr, ks = self.degrade(batch['inp'], scale, random=True, state=state, norm=True) # bn, c, h, w
		lr_blur, lr_clean = torch.split(lr, [self.bs, self.bs], dim=0)
		return lr_blur, lr_clean, ks

	def data(self, batch1, pool):
		# online preprocess: div2k
		lr_blur, lr_clean, ks = self.preprocess(batch1, 'blur+down+kernel')	
		lr_gt_kernel = utils.gt_lr_kernel(lr_blur, lr_clean, ksize=ks)
		p = {'lr': lr_blur, 'gt': batch1['gt'], 'cell': batch1['cell'],\
			 'coord': batch1['coord'], 'lr_gt_kernel': lr_gt_kernel, 'scale': (1/batch1['scale']).type(torch.cuda.FloatTensor)}            
		lr, gt, cell, coord, scale, kernel = pool(p)
		return lr, gt, cell, coord, scale, kernel
			
	def val(self, eval_bsize=None):
		self.model.eval()
		metric_fn = utils.calc_psnr
		val_res = utils.Averager()
		for batch1 in tqdm(self.val_loader1, leave=False, desc='val'):
			for k1, v1 in batch1.items(): 
				batch1[k1] = v1.to(self.device)
				
			lr, gt, cell, coord, scale, _ = self.data(batch1, self.pool_val)
			scale = scale[:, None, :].expand(-1, cell.shape[1],-1)
			with torch.no_grad():
				if eval_bsize is None:
					pred = self.model(lr, coord, cell)   
				else:
					inp = (lr - self.inp_sub) / self.inp_div
					pred = test.batched_predict(self.model, inp, coord, cell, eval_bsize)
					pred = pred*self.gt_div+self.gt_sub
					pred.clamp_(0, 1)
			res = metric_fn(pred, gt)
			val_res.add(res.item(), self.bs)
		return val_res.item()

	def train(self, epoch, timer):
		## for DDP, epoch will go random
		if self.args.DDP:
			self.train_sampler1.sampler.set_epoch(epoch)
			self.val_sampler1.sampler.set_epoch(epoch)
			
		# initial
		if not self.args.DDP or self.local_rank == 0:
			t_epoch_start = timer.t()
		losses = utils.Averager() 

		# train model
		self.model.train()

		for batch1 in tqdm(self.train_loader1, leave=False, desc='train'):
			for k1, v1 in batch1.items(): 
				batch1[k1] = v1.to(self.device)
				
			lr, gt, cell, coord, scale, kernel = self.data(batch1, self.pool)
			
			self.optimizer.zero_grad()
			pred_rgb, pred_lr, pred_k = self.model(lr, coord, cell, state='train')
			loss = self.criterion(pred_rgb, gt)+0.1*self.criterion(pred_lr, lr.detach())
			losses.add(loss.item())
			loss.backward()
			self.optimizer.step()
			
		# lr_scheduler      
		if self.lr_scheduler is not None:
			self.lr_scheduler.step()

		
		if not self.args.DDP or self.local_rank==0:
			# timer stop    
			t = timer.t()
			prog = (epoch -1 + 1) / (self.config['epoch_max'] -1 + 1)
			t_epoch = utils.time_text(t - t_epoch_start)
			t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)

			# validation
			if (self.epoch_val is not None) and (epoch % self.epoch_val == 0):
				val_res = self.val(self.config.get('eval_bsize'))
					
			if val_res > self.max_val_v:
				self.max_val_v = val_res
				utils.save(self.model, self.optimizer, self.config['optimizer'], epoch, self.args.DDP, self.args.savepath, 'best')
			
			# write log
			res = 'epoch {}/{}, loss:{:.4f}, val: psnr={:.4f}, {} {}/{}\n' \
				.format(epoch, self.config['epoch_max'], losses.item(), val_res, t_epoch, t_elapsed, t_all)
			tqdm.write(res)

			# save file
			utils.save(self.model, self.optimizer, self.config['optimizer'], epoch, self.args.DDP, self.args.savepath, 'last')
			if (self.epoch_save is not None) and (epoch % self.epoch_save == 0):
				utils.save(self.model, self.optimizer, self.config['optimizer'], epoch, self.args.DDP, self.args.savepath, 'hun')
			


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', help='config file path')
	parser.add_argument('--savedir', default="your path to save model directory")
	parser.add_argument('--savepath', default=None)
	parser.add_argument('--tag', default=None)
	parser.add_argument('--debug', action='store_true')
	parser.add_argument('--DDP', action='store_true') 
	args = parser.parse_args()
	
	
	#### read config file
	with open(args.config, 'r') as f:
		config = yaml.load(f, Loader=yaml.FullLoader)
		print('config loaded.')
		save_name = args.savedir
		if save_name is None:
			save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
		if args.tag is not None:
			args.savepath = os.path.join(save_name, args.tag)
		if os.path.exists(args.savepath) is False:
			os.makedirs(args.savepath)
			print('{} succeed'.format(args.savepath))
	torch.manual_seed(config['seed'])
	

	#### log file    
	with open(os.path.join(args.savepath, 'config.yaml'), 'w') as f:
		yaml.dump(config, f, sort_keys=False)
	
	t = Trainer(config, args)
	# train epochs
	timer = utils.Timer()
	for epoch in range(1, config['epoch_max']+1):
		t.train(epoch, timer)
	