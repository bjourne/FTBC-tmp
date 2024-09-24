import argparse
import os
import random
import sys

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from datasets import load_dataset

from distributed_utils import get_local_rank, initialize
from models.calibration import bias_corr_model, weights_cali_model, \
    bias_corr_model_update_step_by_step, x_bias_corr_model_update_step_by_step
from models.fold_bn import search_fold_and_remove_bn
from models.ImageNet.models.mobilenet import mobilenetv1
from models.ImageNet.models.resnet import res_spcials, resnet34_snn
from models.ImageNet.models.vgg import vgg16, vgg16_bn, vgg_specials
from models.spiking_layer import SpikeModel, get_maximum_activation

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


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=val_sampler)
    return train_loader, val_loader


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def validate_model(test_loader, ann, num_sample_iters=None):
    correct = 0
    total = 0
    num_sample_iters = len(test_loader) if num_sample_iters is None else num_sample_iters
    ann.eval()
    device = next(ann.parameters()).device
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = ann(inputs)
        _, predicted = outputs.max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx > num_sample_iters:
            break
    return 100 * correct / total


@torch.no_grad()
def validate_model_per_step_mode_snn(test_loader, snn, num_sample_iters=None):
    correct = 0
    total = 0

    num_sample_iters = len(test_loader) if num_sample_iters is None else num_sample_iters

    snn.eval()
    device = next(snn.parameters()).device
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        #-------------------------------
        snn.init_membrane_potential()
        outputs = 0
        for i in range(snn.T):
            outputs += snn(inputs)
        #-------------------------------

        _, predicted = outputs.max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())

        if batch_idx > num_sample_iters:
            break

    return 100 * correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--arch', default='vgg16', type=str, help='network architecture', choices=['vgg16', 'res34', 'mobilenet'])
    parser.add_argument('--seed', default=1000, type=int, help='random seed to reproduce results')
    parser.add_argument('--train_batch_size', default=32, type=int, help='train minibatch size')
    parser.add_argument('--val_batch_size', default=32, type=int, help='valid minibatch size')

    parser.add_argument('--calibration_method', type=str, choices=['snn_cali_baseline', 'ours_wo_avg', 'ours_w_avg'], default='snn_cali_baseline', help='Choose calibration method: "snn_cali_baseline" (default), "ours_wo_avg", or "ours_w_avg".')

    parser.add_argument('--baseline_cali_method', default='none', type=str, help='calibration methods', choices=['none', 'light', 'advanced'])
    parser.add_argument('--T', default=16, type=int, help='snn simulation length')
    parser.add_argument('--usebn', action='store_true', help='use batch normalization in ann')
    parser.add_argument('--channel_wise_vth', action='store_true', help='default: False. Enable or disable using channel-wise Vth, store_true means that it is False by default and becomes True when specified')
    parser.add_argument('--num_cali_sample_batches', default=100, type=int, help='number of sample training data batches to do calibration')
    parser.add_argument('--curr_t_alpha', default=0.5, type=float, help='current time step alpha')
    parser.add_argument('--num_cali_iters', default=5, type=int, help='number of calibration iterations')

    try:
        initialize()
        initialized = True
        torch.cuda.set_device(get_local_rank())
    except:
        print('For some reason, your distributed environment is not initialized, this program may run on separate GPUs')
        initialized = False

    args = parser.parse_args()
    print(args)

    seed_all(args.seed)

    use_bn = args.usebn
    sim_length = args.T
    train_loader, test_loader = build_imagenet_data(
                                    train_batch_size=args.train_batch_size,
                                    val_batch_size=args.val_batch_size,
                                    dist_sample=initialized)

    if args.arch == 'vgg16':
        ann = vgg16_bn(pretrained=True) if args.usebn else vgg16(pretrained=True)
    elif args.arch == 'res34':
        ann = resnet34_snn(pretrained=True, use_bn=args.usebn)
    elif args.arch == 'mobilenet':
        ann = mobilenetv1(pretrained=True)
    else:
        raise NotImplementedError

    search_fold_and_remove_bn(ann)
    ann.cuda()

    ann_acc_sample = validate_model(test_loader, ann, num_sample_iters=2)
    print(f"sample ann accuracy: {ann_acc_sample:.2f} %")

    # ann_acc = validate_model(test_loader, ann)
    # print(f"ann accuracy: {ann_acc:.2f} %")

    snn = SpikeModel(
            model=ann,
            sim_length=sim_length,
            specials=vgg_specials if args.arch == 'vgg16' else res_spcials)
    snn.cuda()

    #--------------------------------------------------------------------------#
    print(f"Start get maximum activation...")
    mse = True # since we always use light and advanced calibration, we set mse=True
    get_maximum_activation(
        train_loader,
        model=snn,
        momentum=0.9,
        iters=5, # iter: number of iterations to calculate the max act
        mse=mse,
        percentile=None,
        sim_length=sim_length,
        channel_wise=args.channel_wise_vth,
        dist_avg=initialized)
    print(f"get maximum activation Done.")

    snn.set_spike_state(use_spike=True)
    snn_acc_sample = validate_model(test_loader, snn, num_sample_iters=2)
    print(f"after get_maximum_activation sample snn accuracy: {snn_acc_sample:.2f} % after baseline SNN Calibration method.") # at {i} iter

    #--------------------------------------------------------------------------#

    best_acc = 0

    if args.calibration_method == 'snn_cali_baseline':
        if args.baseline_cali_method == 'light':
            bias_corr_model(
                model=snn,
                train_loader=train_loader,
                correct_mempot=False,
                dist_avg=initialized)

        if args.baseline_cali_method == 'advanced':
            weights_cali_model(
                model=snn,
                train_loader=train_loader,
                num_cali_samples=1024,
                learning_rate=1e-4,
                dist_avg=initialized)
            bias_corr_model(
                model=snn,
                train_loader=train_loader,
                correct_mempot=True,
                dist_avg=initialized)

        snn.set_spike_state(use_spike=True)
        snn_acc_sample = validate_model(test_loader, snn, num_sample_iters=2)
        print(f"sample snn accuracy: {snn_acc_sample:.2f} % after baseline SNN Calibration method.")

        snn_acc = validate_model(test_loader, snn)
        print(f"snn acc. is {snn_acc:.2f} % after baseline SNN Calibration method")
    elif args.calibration_method == 'ours_wo_avg':
        snn.set_time_dependent_bias()
        for i in range(args.num_cali_iters):
            seed_all(seed=args.seed + i)
            print(f"ours WITHOUT avg calibration iter {i}..."); sys.stdout.flush()
            bias_corr_model_update_step_by_step(
                model=snn, train_loader=train_loader,
                num_cali_sample_batches=args.num_cali_sample_batches,
                curr_t_alpha=args.curr_t_alpha)

            snn.set_spike_state(use_spike=True)
            acc = validate_model_per_step_mode_snn(test_loader, snn, num_sample_iters=2)
            print(f"time_dependent_bias cali WITHOUT avg, sample accuracy {acc:.2f}")

            snn_acc = validate_model_per_step_mode_snn(test_loader, snn)
            print(f"snn acc. is {snn_acc:.2f} % after our method WITHOUT avg")

            if snn_acc > best_acc:
                best_acc = snn_acc
            print(f"best accuracy so far until {i}: {best_acc:.2f} %")

    elif args.calibration_method == 'ours_w_avg':
        snn.set_time_dependent_bias()
        ### avg of sum up to t
        for i in range(args.num_cali_iters):
            seed_all(seed=args.seed + i)
            print(f"ours WITH avg calibration iter {i}..."); sys.stdout.flush()
            x_bias_corr_model_update_step_by_step(
                model=snn, train_loader=train_loader,
                num_cali_sample_batches=args.num_cali_sample_batches,
                curr_t_alpha=args.curr_t_alpha)

            snn.set_spike_state(use_spike=True)
            acc = validate_model_per_step_mode_snn(test_loader, snn, num_sample_iters=2)
            print(f"time_dependent_bias cali WITH avg, sample accuracy {acc:.2f}")

            snn_acc = validate_model_per_step_mode_snn(test_loader, snn)
            print(f"snn acc. is {snn_acc:.2f} % after our method WITH avg")

            if snn_acc > best_acc:
                best_acc = snn_acc
            print(f"best accuracy so far until {i}: {best_acc:.2f} %")
