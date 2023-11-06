import datasets
from datasets.blur import SRMDPreprocessing
from datasets import make

import argparse
import yaml
from tqdm import tqdm
import os
import cv2
import pdb

import torch 
from torchvision.utils import save_image
from torch.utils.data import DataLoader

def save(img, savepath, name):
    save = savepath+'{}'.format(name)
    if os.path.exists(savepath) is False:
        os.makedirs(savepath)
    if os.path.exists(save) is False:
        save_image(img, save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/dataset.yaml")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # read config file
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
    dataset = make(config['dataset'])
    loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=False)
    degrade = SRMDPreprocessing(**config['blur'])
    filenames = sorted(os.listdir(config['dataset']['args']['root_path']))
    
    for image, name in tqdm(zip(loader, filenames), leave=False, desc='generate'):
        lr_blured = degrade(img=image.to(device), scale=config['scale'], random=True, state='gd')
        save(lr_blured, config['savepath'], name)
            
