"""
@author: Shikuang Deng
"""
import torch.nn as nn
import numpy as np

from models.new_relu import th_shift_ReLU
from models.spiking_layer import SPIKE_layer

class VGG16(nn.Module):
    def __init__(
            self,
            modify,
            thresh,
            shift_relu,
            dataset,
            init_epoch
        ):
        super(VGG16, self).__init__()

        self.shift_relu = shift_relu
        self.dataset = dataset
        self.init_epoch = init_epoch
        self.thresh = thresh

        self.max_active = [0] * 16

        ### GROUP 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=(1, 1))
        self.relu_conv1_1 = th_shift_ReLU(simulation_length=self.shift_relu, modify=modify, thresh=self.thresh)

        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1))
        self.relu_conv1_2 = th_shift_ReLU(self.shift_relu, modify, thresh=self.thresh)
        self.maxpool1 = nn.AvgPool2d(2)

        ### GROUP 2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=(1, 1))
        self.relu_conv2_1 = th_shift_ReLU(self.shift_relu, modify, thresh=self.thresh)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=(1, 1))
        self.relu_conv2_2 = th_shift_ReLU(self.shift_relu, modify, thresh=self.thresh)
        self.maxpool2 = nn.AvgPool2d(2)

        ### GROUP 3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=(1, 1))
        self.relu_conv3_1 = th_shift_ReLU(self.shift_relu, modify, thresh=self.thresh)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=(1, 1))
        self.relu_conv3_2 = th_shift_ReLU(self.shift_relu, modify, thresh=self.thresh)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=(1, 1))
        self.relu_conv3_3 = th_shift_ReLU(self.shift_relu, modify, thresh=self.thresh)
        self.maxpool3 = nn.AvgPool2d(2)

        ### GROUP 4
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu_conv4_1 = th_shift_ReLU(self.shift_relu, modify, thresh=self.thresh)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu_conv4_2 = th_shift_ReLU(self.shift_relu, modify, thresh=self.thresh)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=(1, 1))
        self.relu_conv4_3 = th_shift_ReLU(self.shift_relu, modify, thresh=self.thresh)
        self.maxpool4 = nn.AvgPool2d(2)

        ### GROUP 5
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu_conv5_1 = th_shift_ReLU(self.shift_relu, modify, thresh=self.thresh)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu_conv5_2 = th_shift_ReLU(self.shift_relu, modify, thresh=self.thresh)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu_conv5_3 = th_shift_ReLU(self.shift_relu, modify, thresh=self.thresh)
        self.maxpool5 = nn.AvgPool2d(2)

        if self.dataset == 'CIFAR100' or self.dataset == 'CIFAR10':
            in_features = 512 * 2 * 2
        elif self.dataset == 'ImageNet':
            in_features = 512 * 7 * 7

        self.fc1 = nn.Linear(in_features=in_features, out_features=4096)
        self.relu_fc1 = th_shift_ReLU(self.shift_relu, modify, thresh=self.thresh)

        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu_fc2 = th_shift_ReLU(self.shift_relu, modify, thresh=self.thresh)

        if self.dataset == 'CIFAR100':
            self.fc3 = nn.Linear(in_features=4096, out_features=100)
        elif self.dataset == 'CIFAR10':
            self.fc3 = nn.Linear(in_features=4096, out_features=10)
        elif self.dataset == 'ImageNet':
            self.fc3 = nn.Linear(in_features=4096, out_features=1000)

    def renew_max(self, x, y):
        x = max(x, y)
        return x

    def forward(self, x):
        ### GROUP 1
        output = self.conv1_1(x) # x: [N, C, H, W], [128, 3, 224, 224]; output: [128, 64, 224, 224]
        output = self.relu_conv1_1(output) # output: [128, 64, 224, 224]
        self.max_active[0] = self.renew_max(self.max_active[0], output.max())

        output = self.conv1_2(output)
        output = self.relu_conv1_2(output) # [in] output [128, 64, 224, 224],
        self.max_active[1] = self.renew_max(self.max_active[1], output.max())

        output = self.maxpool1(output) # out part of output shape is torch.Size([128, 64, 112, 112])

        # GROUP 2
        output = self.conv2_1(output) # out part of the output is torch.Size([128, 128, 112, 112])
        output = self.relu_conv2_1(output)
        self.max_active[2] = self.renew_max(self.max_active[2], output.max())

        output = self.conv2_2(output)
        output = self.relu_conv2_2(output)
        self.max_active[3] = self.renew_max(self.max_active[3], output.max())
        output = self.maxpool2(output)

        ### GROUP 3
        output = self.conv3_1(output)
        output = self.relu_conv3_1(output)
        self.max_active[4] = self.renew_max(self.max_active[4], output.max())

        output = self.conv3_2(output)
        output = self.relu_conv3_2(output)
        self.max_active[5] = self.renew_max(self.max_active[5], output.max())

        output = self.conv3_3(output)
        output = self.relu_conv3_3(output)
        self.max_active[6] = self.renew_max(self.max_active[6], output.max())
        output = self.maxpool3(output)

        # GROUP 4
        output = self.conv4_1(output)
        output = self.relu_conv4_1(output)
        self.max_active[7] = self.renew_max(self.max_active[7], output.max())

        output = self.conv4_2(output)
        output = self.relu_conv4_2(output)
        self.max_active[8] = self.renew_max(self.max_active[8], output.max())

        output = self.conv4_3(output)
        output = self.relu_conv4_3(output)
        self.max_active[9] = self.renew_max(self.max_active[9], output.max())

        output = self.maxpool4(output)

        ### GROUP 5
        output = self.conv5_1(output)
        output = self.relu_conv5_1(output)

        self.max_active[10] = self.renew_max(self.max_active[10], output.max())
        output = self.conv5_2(output)
        output = self.relu_conv5_2(output)

        self.max_active[11] = self.renew_max(self.max_active[11], output.max())
        output = self.conv5_3(output)
        output = self.relu_conv5_3(output)

        self.max_active[12] = self.renew_max(self.max_active[12], output.max())

        output = self.maxpool5(output)
        output = output.view(x.size(0), -1)

        output = self.fc1(output)
        output = self.relu_fc1(output)
        self.max_active[13] = self.renew_max(self.max_active[13], output.max())

        output = self.fc2(output)
        output = self.relu_fc2(output)
        self.max_active[14] = self.renew_max(self.max_active[14], output.max())

        output = self.fc3(output)
        self.max_active[15] = self.renew_max(self.max_active[15], output.max())

        return output

    def record(self):
        return np.array(self.max_active)

    def load_max_active(self, mat):
        self.max_active = mat


class VGG16_spiking(nn.Module):
    def __init__(self, thresh_list, model, T, shift_snn):
        super(VGG16_spiking, self).__init__()
        self.T = T
        self.time_based_bias = True

        ### group1
        self.conv1_1 = SPIKE_layer(T=T, thresh=thresh_list[0], Conv2d=model.conv1_1, shift_snn=shift_snn)
        self.conv1_2 = SPIKE_layer(T=T, thresh=thresh_list[1], Conv2d=model.conv1_2, shift_snn=shift_snn)
        self.pool1 = nn.AvgPool2d(2)
        ### group2
        self.conv2_1 = SPIKE_layer(T=T, thresh=thresh_list[2], Conv2d=model.conv2_1, shift_snn=shift_snn)
        self.conv2_2 = SPIKE_layer(T=T, thresh=thresh_list[3], Conv2d=model.conv2_2, shift_snn=shift_snn)
        self.pool2 = nn.AvgPool2d(2)
        ### group3
        self.conv3_1 = SPIKE_layer(T=T, thresh=thresh_list[4], Conv2d=model.conv3_1, shift_snn=shift_snn)
        self.conv3_2 = SPIKE_layer(T=T, thresh=thresh_list[5], Conv2d=model.conv3_2, shift_snn=shift_snn)
        self.conv3_3 = SPIKE_layer(T=T, thresh=thresh_list[6], Conv2d=model.conv3_3, shift_snn=shift_snn)
        self.pool3 = nn.AvgPool2d(2)
        ### group4
        self.conv4_1 = SPIKE_layer(T=T, thresh=thresh_list[7], Conv2d=model.conv4_1, shift_snn=shift_snn)
        self.conv4_2 = SPIKE_layer(T=T, thresh=thresh_list[8], Conv2d=model.conv4_2, shift_snn=shift_snn)
        self.conv4_3 = SPIKE_layer(T=T, thresh=thresh_list[9], Conv2d=model.conv4_3, shift_snn=shift_snn)
        self.pool4 = nn.AvgPool2d(2)
        ### group5
        self.conv5_1 = SPIKE_layer(T=T, thresh=thresh_list[10], Conv2d=model.conv5_1, shift_snn=shift_snn)
        self.conv5_2 = SPIKE_layer(T=T, thresh=thresh_list[11], Conv2d=model.conv5_2, shift_snn=shift_snn)
        self.conv5_3 = SPIKE_layer(T=T, thresh=thresh_list[12], Conv2d=model.conv5_3, shift_snn=shift_snn)
        self.pool5 = nn.AvgPool2d(2)

        self.fc1 = SPIKE_layer(T=T, thresh=thresh_list[13], Conv2d=model.fc1, shift_snn=shift_snn)
        self.fc2 = SPIKE_layer(T=T, thresh=thresh_list[14], Conv2d=model.fc2, shift_snn=shift_snn)
        self.fc3 = SPIKE_layer(T=T, thresh=thresh_list[15], Conv2d=model.fc3, shift_snn=shift_snn)

    def set_time_based_bias(self):
        self.conv1_1.set_time_based_bias()
        self.conv1_2.set_time_based_bias()
        self.conv2_1.set_time_based_bias()
        self.conv2_2.set_time_based_bias()
        self.conv3_1.set_time_based_bias()
        self.conv3_2.set_time_based_bias()
        self.conv3_3.set_time_based_bias()
        self.conv4_1.set_time_based_bias()
        self.conv4_2.set_time_based_bias()
        self.conv4_3.set_time_based_bias()
        self.conv5_1.set_time_based_bias()
        self.conv5_2.set_time_based_bias()
        self.conv5_3.set_time_based_bias()
        self.fc1.set_time_based_bias()
        self.fc2.set_time_based_bias()
        self.fc3.set_time_based_bias()

    def init_layer(self):
        self.conv1_1.init_mem()
        self.conv1_2.init_mem()
        self.conv2_1.init_mem()
        self.conv2_2.init_mem()
        self.conv3_1.init_mem()
        self.conv3_2.init_mem()
        self.conv3_3.init_mem()
        self.conv4_1.init_mem()
        self.conv4_2.init_mem()
        self.conv4_3.init_mem()
        self.conv5_1.init_mem()
        self.conv5_2.init_mem()
        self.conv5_3.init_mem()
        self.fc1.init_mem()
        self.fc2.init_mem()
        self.fc3.init_mem()

    def forward(self, x):
        spike_input = x
        output = self.conv1_1(spike_input)
        output = self.conv1_2(output)
        output = self.pool1(output)
        ### group 2
        output = self.conv2_1(output)
        output = self.conv2_2(output)
        output = self.pool2(output)
        ### group 3
        output = self.conv3_1(output)
        output = self.conv3_2(output)
        output = self.conv3_3(output)
        output = self.pool3(output)
        ### group 4
        output = self.conv4_1(output)
        output = self.conv4_2(output)
        output = self.conv4_3(output)
        output = self.pool4(output)
        ### group 5
        output = self.conv5_1(output)
        output = self.conv5_2(output)
        output = self.conv5_3(output)
        output = self.pool5(output)
        output = output.view(x.size(0), -1)

        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output
