import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from datasets import load_dataset
from models.spiking_layer import *
from models.new_relu import *
from models.cnn2 import *
from models.VGG16 import *
from models.ResNet20 import *
from models.CifarNet import *


def dataset(args, data_path):
    batch_size = args.batch_size

    if args.dataset == 'MNIST':
        trans_train = trans_test = transforms.Compose([transforms.ToTensor(), ])
        train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, download=True,
                                                   transform=trans_train)
        test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True,
                                              transform=transforms.ToTensor())

    elif args.dataset == 'CIFAR10':
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True,
                                                     transform=trans_train)
        test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True,
                                                transform=trans_test)
    elif args.dataset == 'CIFAR100':
        trans_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True,
                                                      transform=trans_train)
        test_set = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True,
                                                 transform=trans_test)
    elif args.dataset == 'ImageNet':

        def build_imagenet_data(
                input_size: int = 224,
                train_batch_size: int = 32,
                val_batch_size:int = 32,
                workers: int = 12,
                dist_sample: bool = False
            ):
            """
            val_batch_size = 64 # max!, resnet:300; vgg: 64
            """
            print('==> Using Pytorch Dataset')
            ds = load_dataset("imagenet-1k")
            train_ds = ds["train"]
            val_ds = ds["validation"]

            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            # Apply transformations to the datasets
            class TransformedDataset(Dataset):
                def __init__(self, ds, transform=None):
                    self.ds = ds
                    self.transform = transform
                    self.classes = set(ds['label'])

                def __len__(self):
                    return len(self.ds)

                def __getitem__(self, idx):
                    item = self.ds[idx]
                    image = item['image'].convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    return image, item['label']

            train_transforms = transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])

            val_transforms = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    normalize,
                ])

            train_dataset = TransformedDataset(train_ds, train_transforms)
            val_dataset = TransformedDataset(val_ds, val_transforms)

            if dist_sample:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            else:
                train_sampler = None
                val_sampler = None

            # print(f"train_loader shuffle?: {train_sampler is None}") # True

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=train_batch_size, shuffle=(train_sampler is None),
                num_workers=workers, pin_memory=True, sampler=train_sampler)

            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=val_batch_size, shuffle=False,
                num_workers=workers, pin_memory=True, sampler=val_sampler)
            return train_loader, val_loader

        return build_imagenet_data(train_batch_size=batch_size, val_batch_size=batch_size, workers=12)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=12)
    return train_loader, test_loader


def lr_scheduler(optimizer, epoch):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr_list = [100, 140, 240]
    if epoch in lr_list:
        print('change the learning rate')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model parameters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='CIFAR100', type=str, help='dataset name', choices=['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet'])
    parser.add_argument('--arch', default='VGG16', type=str, help='dataset name', choices=['VGG16', 'ResNet20', 'CIFARNet'])
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='initial learning_rate')
    parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
    parser.add_argument('--thresh', default=0, type=int, help='relu threshold, 0 means use common relu')
    parser.add_argument('--T', default=32, type=int, help='snn simulation length')
    parser.add_argument('--shift_relu', default=0, type=int, help='ReLU shift reference time')
    parser.add_argument('--shift_snn', default=32, type=int, help='SNN left shift reference time')
    parser.add_argument('--step', default=1, type=int, help="record snn output per step, The `args.step` parameter in the provided code snippet is used to determine the frequency at which the output of a Spiking Neural Network (SNN) is recorded during the simulation. Specifically, `args.step` defines the number of simulation steps after which the network's output is sampled and recorded. The variable `simulation_length` is calculated by dividing the total simulation time (`args.T`) by `args.step`, which adjusts the granularity of the simulation's output recording. A smaller value of `args.step` means more frequent sampling and recording of the network's output, leading to a more detailed temporal analysis of the SNN's behavior during the simulation period. This can be crucial for understanding the dynamics of SNNs and for applications where precise timing of neuronal spikes is important.")
    parser.add_argument('--init_epoch', default=-1, type=int, help='use ulimited relu to init parameters')
    parser.add_argument('--num_cali_iters', default=10, type=int, help='number of calibration iterations')
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    data_path = './raw/'
    activation_save_name = args.arch + '_' + args.dataset + '_activation.npy'
    model_save_name = args.arch + '_' + args.dataset + '_state_dict.pth'

    best_acc = 0
    best_epoch = 0

    if args.thresh > 0:
        relu_th = True
    else:
        relu_th = False
    ### use threshold ReLU
    print(args.arch)

    if args.arch == 'VGG16':
        ann = VGG16(
                modify=relu_th,
                shift_relu=args.shift_relu,
                dataset=args.dataset,
                init_epoch=args.init_epoch,
            )
    elif args.arch == 'ResNet20':
        ann = ResNet20(relu_th)
    else:
        ann = CIFARNet(relu_th)
    ann = torch.nn.DataParallel(ann)
    ann.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ann.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    train_loader, test_loader = dataset()
    for epoch in range(num_epochs):
        running_loss = 0
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            ann.train()
            ann.zero_grad()
            optimizer.zero_grad()
            labels = labels.to(device)
            images = images.float().to(device)
            outputs = ann(images, epoch)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                      % (epoch + 1, num_epochs, i + 1, len(train_loader), running_loss))
                running_loss = 0
                print('Time elasped:', time.time() - start_time)
        correct = 0
        total = 0
        optimizer = lr_scheduler(optimizer, epoch)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                ann.eval()
                inputs = inputs.to(device)
                optimizer.zero_grad()
                targets = targets.to(device)
                outputs = ann(inputs, epoch)
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
                total += float(targets.size(0))
                correct += float(predicted.eq(targets).sum().item())
                if batch_idx % 100 == 0:
                    acc = 100. * float(correct) / float(total)
                    print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)

        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))

        acc = 100. * float(correct) / float(total)
        if best_acc < acc and epoch > args.init_epoch:
            best_acc = acc
            best_epoch = epoch + 1
            torch.save(ann.module.state_dict(), model_save_name)
            best_max_act = ann.record()
            # np.save(activation_save_name, best_max_act)
        print('best_acc is: ', best_acc, ' find in epoch: ', best_epoch)
        print('Iters:', epoch, '\n\n\n')
