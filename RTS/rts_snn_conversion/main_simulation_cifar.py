import os
import time
import sys
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models.spiking_layer import *
from models.new_relu import *
from models.cnn2 import *
from models.VGG16_cifar import VGG16, VGG16_spiking
from models.ResNet20 import *
from models.CifarNet import *
from main_train import dataset


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ActivationSaverHook:
    """
    This hook can save output of a layer.
    Note that we have to accumulate T times of the output
    if the model spike state is TRUE.
    """

    def __init__(self):
        self.stored_output = None
        self.stored_input = None

    def __call__(self, module, input_batch, output_batch):
        if self.stored_output is None:
            self.stored_output = output_batch.detach().clone()
        else:
            self.stored_output += output_batch.detach().clone()

        if self.stored_input is None:
            self.stored_input = input_batch[0].detach().clone()
        else:
            self.stored_input += input_batch[0].detach().clone()

    def reset(self):
        self.stored_output = None
        self.stored_input = None


class GetLayerInputOutput:
    def __init__(self, model, target_module):
        self.model = model
        self.module = target_module
        self.data_saver = ActivationSaverHook()

    @torch.no_grad()
    def __call__(self, input):
        self.model.eval()
        h = self.module.register_forward_hook(self.data_saver)
        _ = self.model(input)
        h.remove()
        return self.data_saver.stored_input.detach(), self.data_saver.stored_output.detach()

def emp_bias_corr_moving_avg_timestep(
        ann, module_ann,
        snn, module_snn,
        t, # current time step
        T, # total time step
        train_data: torch.Tensor,
        curr_t_alpha: float = 0.9,
    ):

    device = next(ann.parameters()).device

    ann_get_out = GetLayerInputOutput(ann, module_ann)
    org_out = ann_get_out(train_data)[1]
    ann_get_out.data_saver.reset()

    snn_get_out = GetLayerInputOutput(snn, module_snn)
    snn_out = snn_get_out(train_data)[1]
    snn_get_out.data_saver.reset()
    if len(org_out.shape) == 4:
        ann_mean = org_out.mean(dim=[0, 2, 3])
        snn_mean = snn_out.mean(dim=[0, 2, 3])
    elif len(org_out.shape) == 2:
        ann_mean = org_out.mean(dim=0)
        snn_mean = snn_out.mean(dim=0)


    time_based_bias = (snn_mean - ann_mean).detach()

    time_based_bias1 = module_snn.time_based_biases[t].to(device=device) + curr_t_alpha * time_based_bias
    if module_snn.mem.dim() == 4:
        reshaped_bias = time_based_bias1.view(1, -1, 1, 1)
        reshaped_bias_old = module_snn.time_based_biases[t].view(1, -1, 1, 1)
    elif module_snn.mem.dim() == 2:
        reshaped_bias = time_based_bias1.view(1, -1)
        reshaped_bias_old = module_snn.time_based_biases[t].view(1, -1)

    module_snn.mem += snn_out
    module_snn.mem += reshaped_bias_old.to(device)

    module_snn.mem -= reshaped_bias

    snn_out_new = module_snn.mem.ge(module_snn.thresh).float() * module_snn.thresh
    module_snn.mem -= snn_out_new

    module_snn.time_based_biases[t] = module_snn.time_based_biases[t].to(device=device) + curr_t_alpha * time_based_bias


def bias_corr_model(
        args,
        ann,
        snn,
        T,
        train_loader: torch.utils.data.DataLoader,
        curr_t_alpha=0.5,
    ):
    print(f"==> Start Bias Correction...")
    device = next(ann.parameters()).device

    if args.arch == 'VGG16':
        ### For imagenet-vgg16 and cifar100-vgg16
        ### ann_name : snn_name
        layer_mapping = {
            'relu_conv1_1': 'conv1_1',
            'relu_conv1_2': 'conv1_2',
            'relu_conv2_1': 'conv2_1',
            'relu_conv2_2': 'conv2_2',
            'relu_conv3_1': 'conv3_1',
            'relu_conv3_2': 'conv3_2',
            'relu_conv3_3': 'conv3_3',
            'relu_conv4_1': 'conv4_1',
            'relu_conv4_2': 'conv4_2',
            'relu_conv4_3': 'conv4_3',
            'relu_conv5_1': 'conv5_1',
            'relu_conv5_2': 'conv5_2',
            'relu_conv5_3': 'conv5_3',
            'relu_fc1': 'fc1',
            'relu_fc2': 'fc2',
        }
    elif args.arch == 'ResNet20':
        layer_mapping = {
            'ReLU_1': 'pre_process1',
            'ReLU_2': 'pre_process2',
            'ReLU_3': 'pre_process2',
            'layer1.0.ReLU_residual': 'layer1.0.residual1',
            'layer1.0.ReLU': 'layer1.0',
            'layer1.1.ReLU_residual': 'layer1.1.residual1',
            'layer1.1.ReLU': 'layer1.1',
            'layer2.0.ReLU_residual': 'layer2.0.residual1',
            'layer2.0.ReLU': 'layer2.0',
            'layer2.1.ReLU_residual': 'layer2.1.residual1',
            'layer2.1.ReLU': 'layer2.1',
            'layer3.0.ReLU_residual': 'layer3.0.residual1',
            'layer3.0.ReLU': 'layer3.0',
            'layer3.1.ReLU_residual': 'layer3.1.residual1',
            'layer3.1.ReLU': 'layer3.1',
            'layer4.0.ReLU_residual': 'layer4.0.residual1',
            'layer4.0.ReLU': 'layer4.0',
            'layer4.1.ReLU_residual': 'layer4.1.residual1',
            'layer4.1.ReLU': 'layer4.1',
        }

    ann.eval()
    snn.eval()

    for i, (inputs, target) in enumerate(train_loader):
        print(f"using NO.{i} mini-batch data to calibrate..."); sys.stdout.flush()
        inputs = inputs.to(device=device)

        for ann_name, snn_name in layer_mapping.items():
            module_ann = dict(ann.named_modules())[ann_name]
            module_snn = dict(snn.named_modules())[snn_name]

            snn.init_layer()
            for t in range(T):
                emp_bias_corr_moving_avg_timestep(
                    ann, module_ann,
                    snn, module_snn,
                    t, T,
                    inputs, curr_t_alpha)

        if i >= args.num_cali_sample_batches:
            break

def valid_accuracy(ann, data_loader: torch.utils.data.DataLoader):
    device = next(ann.parameters()).device
    test_loader = data_loader
    correct = 0
    total = 0
    ann.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = ann(inputs)
            _, predicted = outputs.max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
        acc = 100. * float(correct) / float(total)
    return acc

def get_max_activation(ann, train_loader: torch.utils.data.DataLoader):
    device = next(ann.parameters()).device
    correct = 0
    total = 0
    ann.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = ann(inputs)
            _, predicted = outputs.max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            if batch_idx > 10:
                break

def validate_snn_accuracy(args, snn, test_loader: torch.utils.data.DataLoader):
    total = 0
    simulation_length = int(args.T / args.step)
    simulation_loader = torch.zeros(1, simulation_length)

    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            #------------------------------------------------------------------#
            snn.init_layer()
            out_spike_sum = 0
            outputs_ls = []

            for t in range(args.T):
                output = snn(inputs)
                out_spike_sum += output
                if (t + 1) % args.step == 0:
                    sub_result = out_spike_sum / (t + 1)
                    outputs_ls.append(sub_result)
            #------------------------------------------------------------------#
            total += float(targets.size(0))

            for i in range(simulation_length):
                _, predicted = outputs_ls[i].max(1)
                simulation_loader[0, i] += float(predicted.eq(targets).sum().item())

            end_time = time.time()

        corr = 100.000 * simulation_loader / total
        Ts = 0

        for i in range(simulation_length):
            Ts = Ts + args.step
            print('simulation length: ', Ts, ' -> corr: ', corr[0, i].data)

        return corr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model parameters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', default=1000, type=int, help='random seed to reproduce results')
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
    parser.add_argument('--curr_t_alpha', default=0.5, type=float, help='current time step alpha')
    parser.add_argument('--num_cali_iters', default=10, type=int, help='number of calibration iterations')
    parser.add_argument('--num_cali_sample_batches', default=100, type=int, help='number of sample training data batches to do calibration')

    args = parser.parse_args()
    print(args)

    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    # activation_save_name = args.arch + '_' + args.dataset + '_activation.npy'
    # model_save_name = args.arch + '_' + args.dataset + '_state_dict.pth'

    data_path = './raw/'  # dataset path, only useful for CIFAR10/100
    train_loader, test_loader = dataset(args, data_path)

    if args.thresh > 0:
        relu_th = True
    else:
        relu_th = False

    if args.arch == 'VGG16':
        ann = VGG16(
                modify=relu_th,
                thresh=args.thresh,
                shift_relu=args.shift_relu,
                dataset=args.dataset,
                init_epoch=args.init_epoch,
            )
    elif args.arch == 'ResNet20':
        ann = ResNet20(
                dropout=relu_th,
                dataset=args.dataset,
                thresh=args.thresh,
                shift_relu=args.shift_relu,
            )
    else:
        ann = CIFARNet(relu_th)
    ann.to(device)

    model_save_name = f"ckpts/{args.dataset}/{args.arch.lower()}/{args.arch.lower()}-{args.dataset.lower()}-relu{args.thresh}.pth"
    print(f"model_save_name: {model_save_name}. (If not right, changet the ckpt filename)")

    if torch.cuda.is_available():
        map_location = None
    else:
        map_location = torch.device('cpu')
    pretrained_dict = torch.load(model_save_name, map_location=map_location)
    ann.load_state_dict(pretrained_dict)

    if args.arch == 'ResNet20':
        ann = ResNet20x(ann, dataset=args.dataset)

    ann_acc = valid_accuracy(ann, test_loader)

    get_max_activation(ann, train_loader)

    if args.arch == 'VGG16':
        print(ann.max_active)
        print(f"max activation value of VGG16: {ann.max_active}")
        snn = VGG16_spiking(
                thresh_list=ann.max_active,
                model=ann,
                T=args.T,
                shift_snn=args.shift_snn)
    elif args.arch == 'ResNet20':
        snn = ResNet20spike(
                ann,
                dataset=args.dataset,
                T=args.T,
                step=args.step,
                shift_snn=args.shift_snn)
    else:
        snn = CIFARNet_spiking(ann.thresh_list, ann)
    snn.to(device)

    #--------------------------------------------------------------------------#
    baseline_snn_acc_all_T = validate_snn_accuracy(args, snn, test_loader)
    simulation_length = int(args.T / args.step)
    Ts = 0
    for i in range(simulation_length):
        Ts = Ts + args.step
        print(f"\033[31mBaseline SNN accuracy: simulation length: {Ts}, corr: {baseline_snn_acc_all_T[0, i].data}\033[0m")
    #--------------------------------------------------------------------------#
    snn.set_time_based_bias()
    for i in range(args.num_cali_iters):
        print(f"iter {i} calibration")
        seed_all(args.seed + i)
        bias_corr_model(
            args,
            ann=ann,
            snn=snn,
            T=args.T,
            train_loader=train_loader,
            curr_t_alpha=args.curr_t_alpha,
        )

        validate_snn_accuracy(args, snn, test_loader)
