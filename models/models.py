import copy
import torch
import pdb

from utils import freeze_unfreeze

models = {}

def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=None, freeze=False, key=None):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    
    if load_sd:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(load_sd, map_location = device)['model']
        if key:
          state_dict = state_dict[key] 
        model.load_state_dict(state_dict)
    if freeze:
        model = freeze_unfreeze(model, 'freeze')
    return model
