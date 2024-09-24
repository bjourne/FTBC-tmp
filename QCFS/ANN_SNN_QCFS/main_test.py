# import debugpy; debugpy.listen(5678); debugpy.wait_for_client(); # debugpy.breakpoint()

import argparse
import os
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from Models import modelpool
from Preprocess import datapool
from utils import train, val, seed_all, get_logger
from Models.layer import *

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-j','--workers',default=16, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=200, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# model configuration
parser.add_argument('-data', '--dataset',default='cifar100',type=str,help='dataset: cifa10, cifar100, imagenet')
parser.add_argument('-arch', '--model',default='vgg16',type=str,help='model: resnet34')
parser.add_argument('-id', '--identifier', type=str,help='model statedict identifier')

# test configuration
parser.add_argument('-dev','--device',default='0',type=str,help='device')
parser.add_argument('-T', '--time', default=0, type=int, help='snn simulation time')
parser.add_argument('-L', '--L', default=8, type=int, help='Step L')
args = parser.parse_args()

def main():
    global args
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    # preparing data
    train_loader, test_loader = datapool(args.dataset, args.batch_size, dist_sample=False)
    # preparing model
    model = modelpool(args.model, args.dataset)

    model_dir = '%s-checkpoints'% (args.dataset)
    state_dict = torch.load(os.path.join(model_dir, args.identifier + '.pth'), map_location=torch.device('cpu'))

    # if old version state_dict
    keys = list(state_dict.keys())
    for k in keys:
        if "relu.up" in k:
            state_dict[k[:-7]+'act.thresh'] = state_dict.pop(k)
        elif "up" in k:
            state_dict[k[:-2]+'thresh'] = state_dict.pop(k)

    model.load_state_dict(state_dict)

    model.to(device)

    model.set_T(args.time)
    model.set_L(args.L)

    # for m in model.modules():
    #     if isinstance(m, IF):
    #         print(m.thresh)

    acc = val(model, test_loader, args.time, device, sample_iter=10)
    print(acc)


if __name__ == "__main__":
    main()
