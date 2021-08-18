""""
This script contains the model trained and evaluated for the
task of Singing Language Identification. The components of this
model have been refactored from https://github.com/locuslab/TCN
"""


import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x, pad_right=True):
        return x[:, :, :-self.chomp_size].contiguous() if pad_right else x[:, :, self.chomp_size:]


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1, self.dropout1, self.conv2,
            self.chomp2,
            self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, stride, dropout):
        super(TemporalConvNet, self).__init__()
        self.network = nn.Sequential()
        for i, nch in enumerate(num_channels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size-1) * dilation
            self.network.add_module('tblock'+str(i),TemporalBlock(
                n_inputs=in_channels,
                n_outputs=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                dropout=dropout))

    def forward(self, x):
        return self.network(x)



class TCN_SLID(nn.Module):
    def __init__(self, size_in, size_out, list_conv_depths, size_kernel, stride, dropout):
        super(TCN_SLID, self).__init__()
        self.tcn = TemporalConvNet(
            num_inputs=size_in,
            num_channels=list_conv_depths,
            kernel_size=size_kernel,
            stride=stride,
            dropout=dropout)
        self.linear1 = nn.Linear(list_conv_depths[-1], list_conv_depths[-1]//2)
        self.linear2 = nn.Linear(list_conv_depths[-1]//2, list_conv_depths[-1]//4)
        self.linear3 = nn.Linear(list_conv_depths[-1]//4, size_out)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2))
        output = self.linear1(output.transpose(1,2))#.double()
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.softmax(output)

        return output.transpose(1,2)
