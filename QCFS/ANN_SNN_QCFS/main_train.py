import argparse
import os
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim

from os.path import join
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from Models import modelpool
from Preprocess import datapool
from utils import val, seed_all, get_logger

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument(
    '-j','--workers', default=4, type=int,metavar='N',help='number of data loading workers'
)
parser.add_argument(
    '-b','--batch_size', default=300, type=int,metavar='N',help='mini-batch size'
)
parser.add_argument('--seed', default=42, type=int, help='seed for initializing training. ')
parser.add_argument('-suffix','--suffix', default='', type=str,help='suffix')
parser.add_argument('-T', '--time', default=0, type=int, help='snn simulation time')

# model configuration
parser.add_argument('-data', '--dataset',default='cifar100',type=str,help='dataset')
parser.add_argument('-arch','--model',default='vgg16',type=str,help='model')

# training configuration
parser.add_argument('--epochs',default=300,type=int,metavar='N',help='number of total epochs to run')
# 0.05 for cifar100 / 0.1 for cifar10
parser.add_argument(
    '-lr', '--lr',
    default=0.1,
    type=float,
    metavar='LR',
    help='initial learning rate'
)
parser.add_argument('-wd','--weight_decay',default=5e-4, type=float, help='weight_decay')
parser.add_argument('-dev','--device',default='0',type=str,help='device')
parser.add_argument('-L', '--L', default=8, type=int, help='Step L')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(net, device, train_loader, criterion, optimizer, T):
    running_loss = 0
    net.train()
    total = 0
    correct = 0
    for i, (xs, labels) in enumerate((train_loader)):
        print(i)
        optimizer.zero_grad()
        labels = labels.to(device)
        xs = xs.to(device)
        if T > 0:
            outputs = net(xs).mean(0)
        else:
            outputs = net(xs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())

    return running_loss, 100 * correct / total

def main():
    global args
    seed_all(args.seed)

    l_tr, l_te = datapool(args.dataset, args.batch_size)

    model = modelpool(args.model, args.dataset)
    model.set_L(args.L)

    log_dir = '%s-checkpoints'% (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model.to(device)

    crit = CrossEntropyLoss().to(device)

    opt = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs)
    best_acc = 0

    identifier = args.model

    identifier += '_L[%d]'%(args.L)

    if not args.suffix == '':
        identifier += '_%s'%(args.suffix)

    logger = get_logger(join(log_dir, '%s.log'%(identifier)))

    print("Starting", device)

    for epoch in range(args.epochs):
        loss, acc = train(model, device, l_tr, crit, opt, args.time)
        logger.info(
            'Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(
                epoch , args.epochs, loss, acc
            )
        )
        scheduler.step()
        tmp = val(model, l_te, args.time, device)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}\n'.format(epoch , args.epochs, tmp))

        if best_acc < tmp:
            best_acc = tmp
            torch.save(model.state_dict(), join(log_dir, '%s.pth'%(identifier)))

    logger.info('Best Test acc={:.3f}'.format(best_acc))

if __name__ == "__main__":
    main()
