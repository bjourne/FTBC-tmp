"""
@author: Shikuang Deng
"""
import torch
import torch.nn as nn


# spike layer, requires nn.Conv2d (nn.Linear) and thresh
class SPIKE_layer(nn.Module):
    def __init__(
            self,
            thresh,
            Conv2d,
            T,
            shift_snn,
        ):
        super(SPIKE_layer, self).__init__()
        self.t = 0 # temp variable to store the current time step
        self.T = T
        self.shift_snn = shift_snn
        self.thresh = thresh

        self.ops = Conv2d

        self.mem = 0
        if self.shift_snn > 0:
            self.shift = self.thresh / (2 * self.shift_snn)
        else:
            self.shift = 0
        # self.shift_operation()

        self.enable_time_based_bias = False
        self.initialize_time_based_biases(conv=Conv2d)


    def initialize_time_based_biases(self, conv):
        """ Initialize time-dependent biases based on the type of layer. """
        if isinstance(conv, nn.Conv2d):
            bias_shape = (conv.out_channels,)
        elif isinstance(conv, nn.Linear):
            bias_shape = (conv.out_features,)
        else:
            raise NotImplementedError("Unsupported layer type for time-dependent biases")

        # Initialize the biases for each time step
        self.time_based_biases = [torch.zeros(bias_shape) for _ in range(self.T)]

    def set_time_based_bias(self):
        """ Set the time-dependent biases for the module. """
        self.enable_time_based_bias = True

    def init_mem(self):
        self.mem = 0
        self.t = 0

    def shift_operation(self):
        for key in self.state_dict().keys():
            if 'bias' in key:
                pa = self.state_dict()[key]
                pa.copy_(pa + self.shift)

    def forward(self, input):
        x = self.ops(input) + self.shift
        self.mem += x

        #----------------------------------------------------------------------#
        if self.enable_time_based_bias:
            time_based_bias = self.time_based_biases[self.t]
            if self.mem.dim() == 4:
                reshaped_bias = time_based_bias.view(1, -1, 1, 1)
            elif self.mem.dim() == 2:
                reshaped_bias = time_based_bias.view(1, -1)
            else:
                raise RuntimeError("Unexpected shape of self.mem")

            if self.mem.device != reshaped_bias.device:
                reshaped_bias = reshaped_bias.to(self.mem.device)
            self.mem -= reshaped_bias
        #----------------------------------------------------------------------#

        spike = self.mem.ge(self.thresh).float() * self.thresh
        self.mem -= spike
        self.t += 1
        return spike

