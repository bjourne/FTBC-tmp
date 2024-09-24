import argparse
import os
import random
from unittest import result

import numpy as np
import torch

from main_train_cifar import build_data
from models.calibration import bias_corr_model, bias_corr_model_update_step_by_step
from models.CIFAR.models.resnet import res_specials
from models.CIFAR.models.resnet import resnet20 as resnet20_cifar
from models.CIFAR.models.mobilenet import MobileNet
from models.CIFAR.models.resnet import resnet32 as resnet32_cifar
from models.CIFAR.models.vgg import VGG
from models.fold_bn import search_fold_and_remove_bn
from models.spiking_layer import SpikeModel, get_maximum_activation

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
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
def validate_model_per_step_mode_snn(test_loader, ann, num_sample_iters=None):
    """
    For SNN validation
    """
    correct = 0
    total = 0

    num_sample_iters = len(test_loader) if num_sample_iters is None else num_sample_iters

    ann.eval()
    device = next(ann.parameters()).device
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        #-------------------------------
        ann.init_membrane_potential()
        outputs = 0
        for i in range(ann.T):
            outputs += ann(inputs)
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

    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name', choices=['CIFAR10', 'CIFAR100'])
    parser.add_argument('--arch', default='VGG16', type=str, help='network architecture', choices=['VGG16', 'res20', 'mobilenet'])
    parser.add_argument('--dpath', default='./datasets', type=str, help='dataset directory')
    parser.add_argument('--model', default='', type=str, help='model path')
    parser.add_argument('--seed', default=1000, type=int, help='random seed to reproduce results')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--device', default='', type=str, help='device select')
    parser.add_argument('--T', default=16, type=int, help='snn simulation length')
    parser.add_argument('--usebn', action='store_true', help='use batch normalization in ann')
    parser.add_argument('--channel_wise_vth', action='store_true', help='default is False. Enable or disable using channel-wise Vth, store_true means that it is False by default and becomes True when specified')

    parser.add_argument('--calibration_method', type=str, choices=['snn_cali_baseline', 'ours_wo_avg', 'ours_w_avg'], default='snn_cali_baseline', help='Choose calibration method: "snn_cali_baseline" (default), "ours_wo_avg", or "ours_w_avg".')
    parser.add_argument('--baseline_cali_method', default='none', type=str, help='calibration methods', choices=['none', 'light', 'advanced'])

    parser.add_argument('--num_cali_sample_batches', default=10, type=int, help='number of sample training data batches to do calibration')
    parser.add_argument('--curr_t_alpha', default=0.5, type=float, help='current time step alpha')
    parser.add_argument('--num_cali_iters', default=5, type=int, help='number of iterations to calculate the max act')

    args = parser.parse_args()
    print(args)

    seed_all(args.seed)

    use_bn = args.usebn
    sim_length = args.T
    device = args.device
    if args.device == '':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cifar10 = args.dataset == 'CIFAR10'
    train_loader, test_loader = build_data(
                                    batch_size=args.batch_size,
                                    cutout=True,
                                    use_cifar10=use_cifar10,
                                    auto_aug=True,
                                    dpath=args.dpath,
                                )

    if args.arch == 'VGG16':
        ann = VGG('VGG16', use_bn=use_bn, num_class=10 if use_cifar10 else 100)
    elif args.arch == 'res20':
        ann = resnet20_cifar(use_bn=use_bn, num_classes=10 if use_cifar10 else 100)
    elif args.arch == 'mobilenet':
        ann = MobileNet(num_classes=10 if use_cifar10 else 100)
    elif args.arch == 'res32':
        ann = resnet32_cifar(use_bn=use_bn, num_classes=10 if use_cifar10 else 100)
    else:
        raise NotImplementedError

    load_path = 'raw/' + args.dataset + '/' + args.arch + '_wBN_wd5e4_state_dict_best.pth' if use_bn else \
                'raw/' + args.dataset + '/' + args.arch + '_woBN_wd1e4_state_dict_best.pth'
    if args.model != '':
        load_path = args.model
    state_dict = torch.load(load_path, map_location=device)
    ann.load_state_dict(state_dict, strict=True)

    search_fold_and_remove_bn(ann)
    ann.cuda()

    ann_acc = validate_model(test_loader, ann)
    print(f"ANN Accuracy: {ann_acc:.2f}%")

    snn = SpikeModel(model=ann, sim_length=sim_length, specials=res_specials)
    snn.cuda()

    mse = True ### IMPT!!!
    get_maximum_activation(
        train_loader,
        model=snn,
        momentum=0.9,
        iters=5,
        mse=mse,
        percentile=None,
        sim_length=sim_length,
        channel_wise=args.channel_wise_vth,
    )

    snn.set_spike_state(use_spike=True)
    acc = validate_model(test_loader, snn)
    print(f"accuracy after get_maximum_activation: {acc:.2f}%")

    if args.calibration_method == "snn_cali_baseline":
        print(f"Baseline SNN Calibration method: {args.baseline_cali_method} is used")
        if args.baseline_cali_method == 'light':
            bias_corr_model(
                model=snn,
                train_loader=train_loader,
                correct_mempot=False,
                num_cali_sample_batches=1,
            )
        if args.baseline_cali_method == 'advanced':
            # For CIFAR10/100 dataset, calibrating the potential only achieves best results.
            # weights_cali_model(model=snn, train_loader=train_loader, num_cali_samples=1024, learning_rate=1e-5)
            bias_corr_model(
                model=snn,
                train_loader=train_loader,
                correct_mempot=True,
                num_cali_sample_batches=1,
            )

        snn.set_spike_state(use_spike=True)
        # sample_acc = validate_model(test_loader, snn, num_sample_iters=10)
        # print(f"Sample accuracy: {sample_acc:.2f}%")
        acc = validate_model(test_loader, snn)
        print(f"Baseline accuracy: {acc:.2f}%")
    elif args.calibration_method == "ours_wo_avg":
        print(f"Ours SNN Calibration method is used")
        snn.set_time_dependent_bias()
        best_accuracy = 0.0
        i_max = 0
        for i in range(args.num_cali_iters):
            seed_all(seed=args.seed + i)
            bias_corr_model_update_step_by_step(
                model=snn, train_loader=train_loader,
                num_cali_sample_batches=args.num_cali_sample_batches,
                curr_t_alpha=args.curr_t_alpha)

            snn.set_spike_state(use_spike=True)
            acc = validate_model_per_step_mode_snn(test_loader, snn)
            print(f"ours method accuracy at [{i}]: {acc:.2f}%")

            if acc > best_accuracy:
                best_accuracy = acc
                i_max = i
            print(f"best accuracy so far is from [{i_max}]: {best_accuracy:.2f} %")

    elif args.calibration_method == "ours_w_avg":
        ...
