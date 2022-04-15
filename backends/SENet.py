from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import math

class SELayer(nn.Module):
    """
    modified using the code from https://github.com/moskomule/senet.pytorch/blob/8cb2669fec6fa344481726f9199aa611f08c3fbd/senet/se_module.py#L4
    """
    def __init__(self, channel, reduction=10):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, pooling, drop_rate, BN=True, relu_type='leaky'):
        super(ConvLayer, self).__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU() if relu_type == 'leaky' else nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SENet(nn.Module):
    def __init__(self, config):
        super(SENet, self).__init__()
        num, p = config['fil_num'], config['drop_rate']
        self.block1 = ConvLayer(1, num, (7, 2, 0), (3, 2, 0), p)
        self.se_layer1 = SELayer(num)
        self.block2 = ConvLayer(num, 2*num, (4, 1, 0), (2, 2, 0), p)
        self.se_layer2 = SELayer(2 * num)
        self.block3 = ConvLayer(2*num, 4*num, (3, 1, 0), (2, 2, 0), p)
        self.se_layer3 = SELayer(4 * num)
        self.block4 = ConvLayer(4*num, 8*num, (3, 1, 0), (2, 2, 0), p)
        self.se_layer4 = SELayer(8 * num)
        self.size = self.test_size()

    def forward(self, x):
        x = self.block1(x)
        x = self.se_layer1(x)
        x = self.block2(x)
        x = self.se_layer2(x)
        x = self.block3(x)
        x = self.se_layer3(x)
        x = self.block4(x)
        x = self.se_layer4(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return x

    def test_size(self):
        case = torch.ones((1, 1, 182, 218, 182))
        output = self.forward(case)
        return output.shape[1]


if __name__ == "__main__":
    config = {}
    config['fil_num'], config['drop_rate'] = 36, 0.2
    senet = SENet(config)
    print(senet.test_size())

