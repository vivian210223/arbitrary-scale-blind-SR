import argparse
import os
from PIL import Image
import pdb
import yaml

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict
from models.controller import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='your path to LR image.png')
    parser.add_argument('--output', default='your path to output SR image.png')
    parser.add_argument('--model', default='your path to model weight.pth')
    parser.add_argument('--config', help='config file path', default='your path to model config.yaml')
    parser.add_argument('--resolution')
    args = parser.parse_args()

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load file
    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))

    with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    
    # load model
    model = models.make(config['model'], args={'config': config}, load_sd=args.model).to(device)
            
    h, w = list(map(int, args.resolution.split(',')))
    coord = make_coord((h, w)).to(device)
    cell = torch.ones_like(coord)
    
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    
    # draw
    model.eval()
    with torch.no_grad():
        pred = batched_predict(model, ((img - 0.5) / 0.5).to(device).unsqueeze(0),
            coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000, baseline=args.baseline)[0]
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()

    transforms.ToPILImage()(pred).save(args.output)
    
    
