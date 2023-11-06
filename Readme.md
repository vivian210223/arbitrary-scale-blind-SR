# Best of Both Worlds: Learning Arbitrary-scale Blind Super-Resolution via Dual Degradation Representations and Cycle-Consistency (WACV 2024)
## Environment
Our code is based on Ubuntu 20.04, pytorch 1.12.0, CUDA 11.8 (NVIDIA RTX 3090 24GB,) and python 3.10.
## Train
### Before training, revise the path in config file according to your setting
	configs/degradation.yaml
	configs/train-div2k/train_SR.yaml
### Install pytorch wavelets
	https://pytorch-wavelets.readthedocs.io/en/latest/readme.html

#### 1. first train with degradation module
``
```
CUDA_VISIBLE_DEVICES=0 python train_degrade.py --config configs/degradation.yaml --tag (your model name) --savedir (your path to model save director) --queue
```
#### 2. then, train with SR module
```
CUDA_VISIBLE_DEVICES=0 python train_SR.py --config configs/train-div2k/train_ours.yaml --tag (your model name) --savedir (your path to model save director)
```

## Test
```
CUDA_VISIBLE_DEVICES=0 python test.py --model_config (your path to your model config.yaml) --model_weight (your path to your model weight.pth) --test_config (path to test.yaml)
```

## generate SR image
```
CUDA_VISIBLE_DEVICES=0 python demo.py --input (path to LR image) --output (path to save the SR image) --model (path to your model weight.pth) --config (path to your model config.yaml)--resolution (height, weight)
```

### Acknowledgements
This code is built on [LIIF](https://github.com/yinboc/liif), [LTE](https://github.com/jaewon-lee-b/lte) and [Simsiam](https://github.com/facebookresearch/simsiam) We thank the authors for sharing their codes.