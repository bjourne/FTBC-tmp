import sys
import time
import copy

import torch
from distributed_utils.dist_helper import allaverage, allreduce

from models.spiking_layer import SpikeModel, SpikeModule, lp_loss

from .CIFAR.models.resnet import SpikeResModule as SpikeResModule_CIFAR
from .ImageNet.models.resnet import SpikeResModule as SpikeResModule_ImageNet

#--------------- Bias Calibration: our, update every step ----------------------

class ActivationSaverHookOneTimeStep:
    """
    This hook can save output of a layer for one time step. 因此不进行累加.
    if the model spike state is TRUE.
    """
    def __init__(self):
        self.stored_output = None
        self.stored_input = None
        self.stored_residual = None

    def __call__(self, module, input_batch, output_batch):
        if self.stored_output is None:
            self.stored_output = output_batch

        if self.stored_input is None:
            self.stored_input = input_batch[0]

        if len(input_batch) == 2:
            if self.stored_residual is None:
                self.stored_residual = input_batch[1].detach()
        else:
            if self.stored_residual is None:
                self.stored_residual = 0

    def reset(self):
        self.stored_output = None
        self.stored_input = None
        self.stored_residual = None


class GetLayerInputOutputOneTimeStep:
    def __init__(self, model: SpikeModel, target_module: SpikeModule):
        self.model = model
        self.module = target_module
        self.data_saver = ActivationSaverHookOneTimeStep()

    @torch.no_grad()
    def __call__(self, input):
        self.model.eval()
        h = self.module.register_forward_hook(self.data_saver)
        _ = self.model(input)

        h.remove()
        return  self.data_saver.stored_input.detach(), \
                self.data_saver.stored_output.detach(), \
                self.data_saver.stored_residual


def emp_bias_corr_one_step(
        model: SpikeModel,
        module: SpikeModule,
        t,
        train_data: torch.Tensor,
        curr_t_alpha: float = 0.5,
    ):
    device = next(model.parameters()).device

    model.set_spike_state(use_spike=False)
    get_out = GetLayerInputOutputOneTimeStep(model, module)

    ann_out = get_out(train_data)[1]
    get_out.data_saver.reset()

    model.set_spike_state(use_spike=True)
    snn_out = get_out(train_data)[1]
    get_out.data_saver.reset()

    ann_mean = ann_out.mean(3).mean(2).mean(0).detach() if len(ann_out.shape) == 4 else ann_out.mean(0).detach()
    snn_mean = snn_out.mean(3).mean(2).mean(0).detach() if len(snn_out.shape) == 4 else snn_out.mean(0).detach()
    this_timestep_bias = (snn_mean - ann_mean).data.detach()

    if module.mem_pot.dim() == 4:
        old_t_reshaped_bias = module.time_dependent_biases[t].view(1, -1, 1, 1)
        this_t_reshaped_bias = this_timestep_bias.view(1, -1, 1, 1)
    elif module.mem_pot.dim() == 2:
        old_t_reshaped_bias = module.time_dependent_biases[t].view(1, -1)
        this_t_reshaped_bias = this_timestep_bias.view(1, -1)
    else:
        raise ValueError("The dimension of the membrane potential is not supported.")

    old_t_reshaped_bias = old_t_reshaped_bias.to(device)
    this_t_reshaped_bias = this_t_reshaped_bias.to(device)

    module.mem_pot += snn_out
    module.mem_pot += old_t_reshaped_bias

    new_t_reshaped_bias = old_t_reshaped_bias + curr_t_alpha * this_t_reshaped_bias
    module.mem_pot -= new_t_reshaped_bias
    tmp_spike = (module.mem_pot > module.threshold).float() * module.threshold
    module.mem_pot -= tmp_spike

    module.time_dependent_biases[t] = module.time_dependent_biases[t].to(device=device) + curr_t_alpha * this_timestep_bias


def emp_bias_corr_one_step_up_until_t(
        model: SpikeModel,
        module: SpikeModule,
        t,
        train_data: torch.Tensor,
        correct_mempot: bool = False,
        dist_avg: bool = False,
        curr_t_alpha: float = 0.1,
    ):
    device = next(model.parameters()).device

    model.set_spike_state(use_spike=False)
    get_out = GetLayerInputOutputOneTimeStep(model, module)

    ann_out = get_out(train_data)[1]
    get_out.data_saver.reset()

    model.set_spike_state(use_spike=True)
    snn_out = get_out(train_data)[1]

    ann_mean = ann_out.mean(3).mean(2).mean(0).detach() if len(ann_out.shape) == 4 else ann_out.mean(0).detach()

    if not hasattr(module, 'snn_out_sum_up_to_t') or t == 0:
        module.snn_out_sum_up_to_t = torch.zeros_like(snn_out).to(device=device)

    module.snn_out_sum_up_to_t += snn_out
    avg_snn_out_up_to_t = module.snn_out_sum_up_to_t / (t + 1)
    avg_snn_out_up_to_t_mean = avg_snn_out_up_to_t.mean(3).mean(2).mean(0).detach() if len(avg_snn_out_up_to_t.shape) == 4 else avg_snn_out_up_to_t.mean(0).detach()
    bias = (avg_snn_out_up_to_t_mean - ann_mean).data.detach()
    return bias


def bias_corr_model_update_step_by_step(
        model: SpikeModel,
        train_loader: torch.utils.data.DataLoader,
        num_cali_sample_batches=10,
        curr_t_alpha=0.5, # weight for the most recent t observation.
    ):
    """
    This function corrects the bias in SNN, by matching the
    activation expectation in some training set samples.
    Here we only sample one batch of the training set。

    :param model: SpikeModel that need to be corrected with bias
    :param train_loader: Training images
    """
    print(f"==> Start Time Based Bias Correction W/O avg ...")
    device = next(model.parameters()).device

    for i, (input, target) in enumerate(train_loader):
        if i > num_cali_sample_batches-1:
            break

        print(f"cali using the sample input index: {i}...")
        input = input.to(device=device)
        for name, module in model.named_modules():
            if isinstance(module, SpikeModule):
                model.init_membrane_potential()
                for t in range(model.T):
                    emp_bias_corr_one_step(model=model, module=module, t=t, train_data=input, curr_t_alpha=curr_t_alpha)


def x_bias_corr_model_update_step_by_step(
        model: SpikeModel,
        train_loader: torch.utils.data.DataLoader,
        num_cali_sample_batches=100,
        curr_t_alpha: float = 0.5,
    ):
    print(f"==> Start Time Based Bias Correction WITH avg ...")
    device = next(model.parameters()).device

    for i, (input, target) in enumerate(train_loader):
        print(f"sample inputs index: {i}...")
        input = input.to(device=device)
        for name, module in model.named_modules():
            if isinstance(module, SpikeModule):
                ### ann mode
                model.set_spike_state(use_spike=False)
                get_out = GetLayerInputOutputOneTimeStep(model, module)
                ann_out = get_out(input)[1]
                get_out.data_saver.reset()
                ann_mean = ann_out.mean(3).mean(2).mean(0).detach() if len(ann_out.shape) == 4 else ann_out.mean(0).detach()

                model.set_spike_state(use_spike=True)
                model.init_membrane_potential()
                for t in range(model.T):
                    snn_out_sum_up_to_t_temp = torch.zeros_like(ann_out).to(device=device).detach()

                    model.init_membrane_potential()
                    for t_temp in range(t+1):
                        snn_out = get_out(input)[1]
                        get_out.data_saver.reset()
                        snn_out_sum_up_to_t_temp += snn_out

                    avg_snn_out_up_to_t = snn_out_sum_up_to_t_temp / (t + 1)
                    avg_snn_out_up_to_t_mean = avg_snn_out_up_to_t.mean(3).mean(2).mean(0).detach() if len(avg_snn_out_up_to_t.shape) == 4 else avg_snn_out_up_to_t.mean(0).detach()
                    bias = (avg_snn_out_up_to_t_mean - ann_mean).data.detach()

                    module.time_dependent_biases[t] = curr_t_alpha * bias + (1-curr_t_alpha) * module.time_dependent_biases[t].to(device=device)
                    module.time_dependent_biases[t] = module.time_dependent_biases[t].to(device=device)

        if i == num_cali_sample_batches:
            break


def bias_corr_model(
        model: SpikeModel,
        train_loader: torch.utils.data.DataLoader,
        correct_mempot: bool = False,
        dist_avg: bool = False,
        num_cali_sample_batches: int = 10,
        ):
    """
    :param model: SpikeModel that need to be corrected with bias
    :param train_loader: Training images
    :param correct_mempot: if True, the correct the initial membrane potential
    :param dist_avg: if True, then average the tensor between distributed GPUs
    :return: SpikeModel with corrected bias
    """
    print(f"==> Start Bias Correction...")
    device = next(model.parameters()).device
    for i, (input, target) in enumerate(train_loader):
        if i > num_cali_sample_batches-1:
            break
        input = input.to(device=device)
        for name, module in model.named_modules():
            if isinstance(module, SpikeModule):
                emp_bias_corr(model, module, input, correct_mempot, dist_avg)


def emp_bias_corr(
        model: SpikeModel,
        module: SpikeModule,
        train_data: torch.Tensor,
        correct_mempot: bool = False,
        dist_avg: bool = False):
    model.set_spike_state(use_spike=False)
    get_out = GetLayerInputOutput(model, module)
    ann_out = get_out(train_data)[1]
    model.set_spike_state(use_spike=True)
    get_out.data_saver.reset()
    snn_out = get_out(train_data)[1]
    snn_out = snn_out / model.T
    if not correct_mempot:
        ann_mean = ann_out.mean(3).mean(2).mean(0).detach() if len(ann_out.shape) == 4 else ann_out.mean(0).detach()
        snn_mean = snn_out.mean(3).mean(2).mean(0).detach() if len(snn_out.shape) == 4 else snn_out.mean(0).detach()
        bias = (snn_mean - ann_mean).data.detach()

        if dist_avg:
            allaverage(bias)
        if module.bias is None:
            module.bias = - bias
        else:
            module.bias.data = module.bias.data - bias
    else:
        ann_mean, snn_mean = ann_out.mean(0, keepdim=True), snn_out.mean(0, keepdim=True)
        pot_init_temp = ((ann_mean - snn_mean) * model.T).data.detach()
        if dist_avg:
            allaverage(pot_init_temp)
        module.mem_pot_init = pot_init_temp


class ActivationSaverHook:
    """
    This hook can save output of a layer.
    Note that we have to accumulate T times of the output
    if the model spike state is TRUE.
    """

    def __init__(self):
        self.stored_output = None
        self.stored_input = None
        self.stored_residual = None

    def __call__(self, module, input_batch, output_batch):
        if self.stored_output is None:
            self.stored_output = output_batch
        else:
            self.stored_output = output_batch + self.stored_output

        if self.stored_input is None:
            self.stored_input = input_batch[0]
        else:
            self.stored_input = input_batch[0] + self.stored_input

        if len(input_batch) == 2:
            if self.stored_residual is None:
                self.stored_residual = input_batch[1].detach()
            else:
                self.stored_residual = input_batch[1].detach() + self.stored_residual
        else:
            if self.stored_residual is None:
                self.stored_residual = 0

    def reset(self):
        self.stored_output = None
        self.stored_input = None
        self.stored_residual = None


class GetLayerInputOutput:
    def __init__(self, model: SpikeModel, target_module: SpikeModule):
        self.model = model
        self.module = target_module
        self.data_saver = ActivationSaverHook()

    @torch.no_grad()
    def __call__(self, input):
        self.model.eval()
        h = self.module.register_forward_hook(self.data_saver)
        _ = self.model(input)
        h.remove()
        return self.data_saver.stored_input.detach(), \
               self.data_saver.stored_output.detach(), \
               self.data_saver.stored_residual

def floor_ste(x):
    return (x.floor() - x).detach() + x

def weights_cali_model(
        model: SpikeModel,
        train_loader: torch.utils.data.DataLoader,
        learning_rate: float = 4e-5,
        optimize_iter: int = 5000,
        batch_size: int = 32,
        num_cali_samples: int = 1024,
        dist_avg: bool = False):
    print('==> Start Weights Calibration...')
    data_sample = []
    for (input, target) in train_loader:
        data_sample += [input]
        if len(data_sample) * data_sample[-1].shape[0] >= num_cali_samples:
            break
    data_sample = torch.cat(data_sample, dim=0)[:num_cali_samples]

    for name, module in model.named_modules():
        if isinstance(module, (SpikeResModule_ImageNet, SpikeResModule_CIFAR)):
            weights_cali_res_layer(model, module, data_sample, learning_rate, optimize_iter,
                                   batch_size, num_cali_samples, dist_avg=dist_avg)
        elif isinstance(module, SpikeModule):
            weights_cali_layer(model, module, data_sample, learning_rate, optimize_iter,
                               batch_size, num_cali_samples, dist_avg=dist_avg)


def weights_cali_layer(
        model: SpikeModel,
        module: SpikeModule,
        data_sample: torch.Tensor,
        learning_rate: float = 2e-5,
        optimize_iter: int = 10000,
        batch_size: int = 32,
        num_cali_samples: int = 1024,
        keep_gpu: bool = True,
        loss_func=lp_loss,
        dist_avg: bool = False):

    get_out = GetLayerInputOutput(model, module)
    device = next(module.parameters()).device
    data_sample = data_sample.to(device)
    cached_batches = []
    for i in range(int(data_sample.size(0) / batch_size)):
        model.set_spike_state(use_spike=False)
        _, cur_out, _ = get_out(data_sample[i * batch_size:(i + 1) * batch_size])
        get_out.data_saver.reset()
        model.set_spike_state(use_spike=True)
        cur_inp, _, _ = get_out(data_sample[i * batch_size:(i + 1) * batch_size])
        get_out.data_saver.reset()
        cached_batches.append((cur_inp.cpu(), cur_out.cpu()))

    cached_inps = torch.cat([x[0] for x in cached_batches])

    cached_inps = cached_inps / model.T
    cached_outs = torch.cat([x[1] for x in cached_batches])

    del cached_batches
    torch.cuda.empty_cache()

    if keep_gpu:
        cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)

    optimizer = torch.optim.Adam([module.weight], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimize_iter, eta_min=0.)

    model.set_spike_state(use_spike=True)
    for i in range(optimize_iter):
        idx = torch.randperm(num_cali_samples)[:batch_size]
        cur_inp = cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        optimizer.zero_grad()
        module.zero_grad()

        snn_out = module.fwd_func(cur_inp, module.weight, module.bias, **module.fwd_kwargs)
        snn_out = torch.clamp(snn_out / module.threshold * model.T, min=0, max=model.T)
        snn_out = floor_ste(snn_out) * module.threshold / model.T
        err = loss_func(snn_out, cur_out)
        err.backward()
        if dist_avg:
            allreduce(module.weight.grad)
        optimizer.step()
        scheduler.step()


def weights_cali_res_layer(model: SpikeModel, module: SpikeModule, data_sample: torch.Tensor,
                           learning_rate: float = 1e-5, optimize_iter: int = 10000,
                           batch_size: int = 8, num_cali_samples: int = 256, keep_gpu: bool = True,
                           loss_func=lp_loss, dist_avg: bool = False):

    get_out = GetLayerInputOutput(model, module)
    device = next(module.parameters()).device
    data_sample = data_sample.to(device)
    cached_batches = []
    for i in range(int(data_sample.size(0) / batch_size)):
        model.set_spike_state(use_spike=False)
        _, cur_out, _ = get_out(data_sample[i * batch_size:(i + 1) * batch_size])
        get_out.data_saver.reset()
        model.set_spike_state(use_spike=True)
        cur_inp, _, cur_res = get_out(data_sample[i * batch_size:(i + 1) * batch_size])
        get_out.data_saver.reset()
        cached_batches.append((cur_inp.cpu(), cur_out.cpu(), cur_res.cpu()))

    cached_inps = torch.cat([x[0] for x in cached_batches])
    cached_inps = cached_inps / model.T
    cached_ress = torch.cat([x[2] for x in cached_batches])
    cached_ress = cached_ress / model.T
    cached_outs = torch.cat([x[1] for x in cached_batches])

    del cached_batches
    torch.cuda.empty_cache()

    if keep_gpu:
        cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)
        cached_ress = cached_ress.to(device)

    optimizer = torch.optim.Adam([module.weight], lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optimize_iter, eta_min=0.)

    model.set_spike_state(use_spike=True)
    for i in range(optimize_iter):
        idx = torch.randperm(num_cali_samples)[:batch_size]
        cur_inp = cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        cur_res = cached_ress[idx].to(device)
        optimizer.zero_grad()
        module.zero_grad()

        snn_out = module.fwd_func(cur_inp, module.weight, module.bias, **module.fwd_kwargs) + cur_res
        snn_out = torch.clamp(snn_out / module.threshold * model.T, min=0, max=model.T)
        snn_out = floor_ste(snn_out) * module.threshold / model.T
        err = loss_func(snn_out, cur_out)
        err.backward()
        if dist_avg:
            allreduce(module.weight.grad)
        optimizer.step()
        scheduler.step()
