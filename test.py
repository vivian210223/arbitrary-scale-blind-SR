import argparse
import yaml
import pdb
from tqdm import tqdm

import datasets
import models
import utils
from models.controller import *
from datasets.blur import SRMDPreprocessing

import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def batched_predict(model, inp, coord, cell, bsize):
	n = coord.shape[1]
	ql = 0
	preds = []
		
	w = utils.wave(inp, model.wav)
	feature = model.encoder(w) # fix
	model.SR.gen_feat(inp, feature)

	while ql < n:
		qr = min(ql + bsize, n)
		pred = model.SR.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
		preds.append(pred)
		ql = qr
	pred = torch.cat(preds, dim=1)
	return pred


def evaluate(loader, model, device, scale=None, data_norm=None, eval_type=None, eval_bsize=None, verbose=False):
	model.eval()
	
	if data_norm is None:
		data_norm = {
			'inp': {'sub': [0], 'div': [1]},
			'gt': {'sub': [0], 'div': [1]}
		}
	t = data_norm['inp']
	inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).to(device)
	inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).to(device)
	t = data_norm['gt']
	gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).to(device)
	gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).to(device)

	val_res_s, val_res_p, val_res_l = utils.Averager(), utils.Averager(), utils.Averager()
	lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)
				
	pbar = tqdm(loader, leave=False, desc='test')
	for batch in pbar:
		for k, v in batch.items(): 
			batch[k] = v.to(device) 

		if eval_bsize is None:
			with torch.no_grad():
				shape = batch['coord'].shape
				pred = model(batch['inp'], batch['coord'], batch['cell'])
					
		else:
			with torch.no_grad():
				inp = (batch['inp'] - inp_sub) / inp_div
				pred = batched_predict(model, inp,
					batch['coord'], batch['cell'], eval_bsize)
				pred = pred * gt_div + gt_sub
				pred.clamp_(0, 1)

		if eval_type is not None: # reshape for shaving-eval
			try:
				shape = [batch['inp'].shape[0], batch['shape'][0,1].item() , batch['shape'][0,2].item(), 3]
				pred = pred.view(*shape) \
					.permute(0, 3, 1, 2).contiguous()
				batch['gt'] = batch['gt'].view(*shape) \
					.permute(0, 3, 1, 2).contiguous()
			except:
				pdb.set_trace()

		with torch.no_grad():
			l = lpips(pred, batch['gt'])
			p = utils.calc_psnr(pred, batch['gt'], dataset='benchmark')
			s = utils.calc_ssim(pred, batch['gt']) 

		val_res_l.add(l.item(), batch['gt'].shape[0])
		val_res_p.add(p.item(), batch['gt'].shape[0])
		val_res_s.add(s.item(), batch['gt'].shape[0])
		if verbose:
			pbar.set_description('psnr: {:.4f}, ssim: {:.4f}, lpips: {:.4f}'\
				.format(val_res_p.item(), val_res_s.item(),  val_res_l.item()))

	return val_res_p.item(), val_res_s.item(), val_res_l.item()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_config', help='model config file path', default='configs/train-div2k/train_liif.yaml')
	parser.add_argument('--model_weight', default='your path to model_weight.pth')
	parser.add_argument('--test_config', help='test config file path', default='configs/test/test-set5-2.yaml')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	with open(args.test_config, 'r') as f:
			testconfig = yaml.load(f, Loader=yaml.FullLoader)
	loader = datasets.make_data_loaders(testconfig, DDP=False, state='test')
		
	with open(args.model_config, 'r') as f:
		modelconfig = yaml.load(f, Loader=yaml.FullLoader)
		
	model = models.make(modelconfig['model'], args={'config': modelconfig}, load_sd=args.model_weight).to(device)
		
	psnr, ssim, lpips = evaluate(loader, model, device,
		data_norm=testconfig.get('data_norm'),
		eval_type=testconfig.get('eval_type'),
		eval_bsize=testconfig.get('eval_bsize'),
		verbose=True)
	print('psnr: {:.4f}, ssim: {:.4f}, lpips: {:.4f}'.format(psnr, ssim, lpips))

