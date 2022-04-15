from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import math

"""
design:

"""

class Model(nn.Module):
    def __init__(self, backbone, mlp, task):
        super(Model, self).__init__()
        self.backbone = backbone
        self.mlp = mlp
        self.softmax = nn.Softmax(dim=1) if task == 'ADD' else None

    def forward(self, x):
        x = self.backbone(x)
        x = self.mlp(x)
        if self.softmax: x = self.softmax(x)
        return x


class ECALayer(nn.Module):
    def __init__(self, channel):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        t = int(abs((math.log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


class MultiModal_CNN_Bone(nn.Module):
    def __init__(self, config, nonImg_size):
        super(MultiModal_CNN_Bone, self).__init__()
        self.nonImg_size = nonImg_size
        num, p = config['fil_num'], config['drop_rate']
        self.block1 = ConvLayer(1, num, (7, 2, 0), (3, 2, 0), p)
        self.block2 = ConvLayer(num, 2*num, (4, 1, 0), (2, 2, 0), p)
        self.se2 = SELayer(2*num, nonImg_size)
        self.block3 = ConvLayer(2*num, 4*num, (3, 1, 0), (2, 2, 0), p)
        self.se3 = SELayer(4*num, nonImg_size)
        self.block4 = ConvLayer(4*num, 8*num, (3, 1, 0), (2, 2, 0), p)
        self.se4 = SELayer(8*num, nonImg_size)
        self.size = self.test_size()

    def forward(self, x, feature):
        x = self.block1(x)
        x = self.block2(x)
        x = x + self.se2(x, feature)
        x = self.block3(x)
        x = x + self.se3(x, feature)
        x = self.block4(x)
        x = x + self.se4(x, feature)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return x

    def test_size(self):
        case = torch.ones((2, 1, 182, 218, 182))
        feature = torch.ones((2, self.nonImg_size))
        output = self.forward(case, feature)
        return output.shape[1]


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, pooling, drop_rate, BN=True, relu_type='leaky'):
        super(ConvLayer, self).__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU() if relu_type=='leaky' else nn.ReLU()
        self.dropout = nn.Dropout(drop_rate) 
       
    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ConvLayer2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, pooling, drop_rate, BN=True, relu_type='leaky'):
        super(ConvLayer2, self).__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv1by1 = nn.Conv3d(in_channels, in_channels//2, 1, 1, 0)
        self.conv = nn.Conv3d(in_channels//2, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN1 = nn.BatchNorm3d(in_channels//2)
        self.BN2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU() if relu_type == 'leaky' else nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv1by1(x)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pooling(x)
        x = self.BN2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class _CNN_Bone(nn.Module):
    def __init__(self, config):
        super(_CNN_Bone, self).__init__()
        num, p = config['fil_num'], config['drop_rate']
        self.block1 = ConvLayer(1, num, (7, 2, 0), (3, 2, 0), p)
        self.block2 = ConvLayer(num, 2*num, (4, 1, 0), (2, 2, 0), p)
        self.block3 = ConvLayer(2*num, 4*num, (3, 1, 0), (2, 2, 0), p)
        self.block4 = ConvLayer(4*num, 8*num, (3, 1, 0), (2, 2, 0), p)
        self.size = self.test_size()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return x

    def test_size(self):
        case = torch.ones((1, 1, 182, 218, 182))
        output = self.forward(case)
        return output.shape[1]


class MLP(nn.Module):
    def __init__(self, in_size, config):  # if binary out_size=2; trinary out_size=3
        super(MLP, self).__init__()
        fil_num, drop_rate, out_size = config['fil_num'], config['drop_rate'], config['out_size']
        self.fil_num = fil_num
        self.out_size = out_size
        self.in_size = in_size
        self.dense1 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_size, fil_num),
        )
        self.dense2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(fil_num, out_size),
        )

    def forward(self, x, get_intermediate_score=False):
        x = self.dense1(x)
        if get_intermediate_score:
            return x
        x = self.dense2(x)
        return x

    def dense_to_conv(self):
        fcn = copy.deepcopy(self)
        A = fcn.dense1[1].weight.view(self.fil_num, self.in_size//(6*6*6), 6, 6, 6)
        B = fcn.dense2[2].weight.view(self.out_size, self.fil_num, 1, 1, 1)
        C = fcn.dense1[1].bias
        D = fcn.dense2[2].bias
        fcn.dense1[1] = nn.Conv3d(self.in_size//(6*6*6), self.fil_num, 6, 1, 0).cuda()
        fcn.dense2[2] = nn.Conv3d(self.fil_num, self.out_size, 1, 1, 0).cuda()
        fcn.dense1[1].weight = nn.Parameter(A)
        fcn.dense2[2].weight = nn.Parameter(B)
        fcn.dense1[1].bias = nn.Parameter(C)
        fcn.dense2[2].bias = nn.Parameter(D)
        return fcn


class MLP2(nn.Module):
    def __init__(self, in_size, feature_size, config):  # if binary out_size=2; trinary out_size=3
        super(MLP2, self).__init__()
        fil_num, drop_rate, out_size = config['fil_num'], config['drop_rate'], config['out_size']
        self.fil_num = fil_num
        self.out_size = out_size
        self.in_size = in_size
        self.dense1 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_size, fil_num),
        )
        self.dense2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(fil_num + feature_size, out_size),
        )

    def forward(self, x, y):
        x = self.dense1(x)
        x = torch.cat((x, y.float()), 1)
        x = self.dense2(x)
        return x


class MLP3(nn.Module):
    def __init__(self, mri_size, nonImg_size, config):
        super(MLP3, self).__init__()
        fil_num, drop_rate, out_size = config['fil_num'], config['drop_rate'], config['out_size']
        self.mri_emb = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(mri_size, fil_num),
            nn.BatchNorm1d(fil_num),
        )
        self.nonImg_emb = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(nonImg_size, fil_num),
            nn.BatchNorm1d(fil_num),
        )
        self.dense1 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(2*fil_num, fil_num),
        )
        self.dense2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(fil_num, out_size),
        )

    def forward(self, mri, nonImg):
        mri = self.mri_emb(mri)
        nonImg = self.nonImg_emb(nonImg.float())
        x = torch.cat((mri, nonImg), 1)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


if __name__ == "__main__":
    pass

