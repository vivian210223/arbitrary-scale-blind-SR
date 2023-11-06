import copy
import pdb

from torch.utils.data.distributed import DistributedSampler 
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data import RandomSampler

from datasets.BatchSamplerRandScale import BatchSamplerRandScale

datasets = {}

def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator

def make(dataset_spec, args=None):
    if args is not None:
        dataset_args = copy.deepcopy(dataset_spec['args'])
        dataset_args.update(args)
    else:
        dataset_args = dataset_spec['args']
    dataset = datasets[dataset_spec['name']](**dataset_args)
    return dataset

def make_data_loader(spec, tag, DDP, bs, rank, ngpu):
    if spec is None:
        return None
    dataset = make(spec['dataset'], args={'rank': rank, 'ngpu': ngpu})
    dataset = make(spec['wrapper'], args={'dataset': dataset})
    sampler = None
    shuffle = (tag == 'train')
    if DDP:
        # Shuffle here (set True) if needed rather than in DataLoader
        sampler = DistributedSampler(dataset)
        shuffle = None  
    elif tag != 'test':
        sampler = RandomSampler(dataset)
        shuffle = None
    
    if not DDP or dist.get_rank() == 0:
        print('{} dataset rank{}: size={}'.format(tag, rank, len(dataset)))
        
    batchsampler = BatchSamplerRandScale(sampler, bs, drop_last=True, scale_range=[1, 4])
    if tag == 'test':
        loader = DataLoader(dataset, batch_size=bs,
         num_workers=0, pin_memory=True)
        return loader
    else:
        loader = DataLoader(dataset, batch_sampler=batchsampler,
            num_workers=0, pin_memory=True)
        return loader, batchsampler

def make_data_loaders(config, DDP, rank=0, ngpu=1, state='SR'):    
    bs = config['batch_size']
    if state == 'SR':
        train_loader1, train_sampler1 = make_data_loader(config.get('train_dataset1'), 'train', DDP, bs, rank, ngpu)
        val_loader1, val_sampler1 = make_data_loader(config.get('val_dataset1'), 'val', DDP, bs, rank, ngpu)
        return train_loader1, train_sampler1, val_loader1, val_sampler1
    elif state =='degrade':
        loader1, sampler1 = make_data_loader(config.get('dataset1'), 'degrade', DDP, bs, rank, ngpu)
        return loader1, sampler1
    elif state == 'test':
        loader = make_data_loader(config.get('test_dataset'), 'test', False, bs, rank, ngpu)
        return loader
    
