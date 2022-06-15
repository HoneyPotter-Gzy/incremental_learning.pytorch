#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2022/6/14 23:58  
@Author: Zheyuan Gu
@File: my_resnet_gzy.py
@Description: None
'''

"""Pytorch port of the resnet used for CIFAR100 by iCaRL.

https://github.com/srebuffi/iCaRL/blob/master/iCaRL-TheanoLasagne/utils_cifar100.py
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from typing import Type, Any, Callable, Union, List, Optional

from inclearn.lib import pooling

logger = logging.getLogger(__name__)


class DownsampleStride(nn.Module):

    def __init__(self, n=2):
        super(DownsampleStride, self).__init__()
        self._n = n

    def forward(self, x):
        return x[..., ::2, ::2]

# 下采样卷积模块
class DownsampleConv(nn.Module):

    def __init__(self, inplanes, planes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, stride=2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        return self.conv(x)

# 残差块
class ResidualBlock(nn.Module):
    expansion = 1
    # TODO: increase_dim到底是什么……（我猜是是否要升维
    def __init__(self, inplanes, increase_dim=False, last_relu=False, downsampling="stride"):
        super(ResidualBlock, self).__init__()

        self.increase_dim = increase_dim
        # 如果要增加维度，那么步长设置为2，输出通道数是原来的2倍
        if increase_dim:
            first_stride = 2
            planes = inplanes * 2
        # 如果不增加维度，那么步长设置为1，输出通道数和原来一样
        else:
            first_stride = 1
            planes = inplanes

        self.conv_a = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=first_stride, padding=1, bias=False
        )
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        if increase_dim:
            if downsampling == "stride":
                self.downsampler = DownsampleStride()
                self._need_pad = True
            else:
                self.downsampler = DownsampleConv(inplanes, planes)
                self._need_pad = False

        self.last_relu = last_relu

    @staticmethod
    def pad(x):
        return torch.cat((x, x.mul(0)), 1)

    def forward(self, x):
        y = self.conv_a(x)
        y = self.bn_a(y)
        y = F.relu(y, inplace=True)

        y = self.conv_b(y)
        y = self.bn_b(y)

        if self.increase_dim:
            x = self.downsampler(x)
            if self._need_pad:
                x = self.pad(x)

        y = x + y

        if self.last_relu:
            y = F.relu(y, inplace=True)

        return y

# make_layer的返回值：Stage(layers, block_relu = self.last_relu)
class Stage(nn.Module):

    def __init__(self, blocks, block_relu=False):
        super().__init__()
        # 一个残差块
        self.blocks = nn.ModuleList(blocks)
        self.block_relu = block_relu

    def forward(self, x):
        intermediary_features = []

        for b in self.blocks:
            x = b(x)
            # 记录中间层特征
            intermediary_features.append(x)

            if self.block_relu:
                x = F.relu(x)
        # 返回中间层特征和最终的输出
        return intermediary_features, x


# class CifarResNet(nn.Module):
class MyResNetGZY(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(
        self,
        layers: List[int],
        nf=64,  # num_features
        channels=1,  # 通道数
        preact=False,
        zero_residual=True,
        pooling_config={"type": "avg"},  # 默认平均池化
        downsampling="stride",  # 下采样：步长
        final_layer=False,
        all_attentions=False,
        last_relu=False,
        **kwargs
    ):
        """ Constructor
        Args:
          depth: number of layers.
          num_classes: number of classes
          base_width: base width
        """
        if kwargs:
            raise ValueError("Unused kwargs: {}.".format(kwargs))

        self.all_attentions = all_attentions
        logger.info("Downsampling type {}".format(downsampling))
        self._downsampling_type = downsampling
        self.last_relu = last_relu

        # Block = ResidualBlock if not preact else PreActResidualBlock
        Block = ResidualBlock

        # super(CifarResNet, self).__init__()
        super(MyResNetGZY, self).__init__()

        # self.conv_1_3x3 = nn.Conv2d(channels, nf, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(channels, nf, kernel_size = (7, 7), stride = (1, 1), padding = 3, bias = False)
        # nf: num_features, 特征个数，函数默认参数是64
        # TODO: 这里要过BN吗
        self.bn_1 = nn.BatchNorm2d(nf)
        # 四个layer，以下的block都是残差块
        # increase_dim是指，n是指Block的个数
        # TODO: 明天起来改写_make_layer
        # 接口make_layer要返回stage类型
        # self.stage_1 = self._make_layer(Block, nf, increase_dim=False, n=n)
        # self.stage_2 = self._make_layer(Block, nf, increase_dim=True, n=n - 1)
        # self.stage_3 = self._make_layer(Block, 2 * nf, increase_dim=True, n=n - 2)
        # 和原模型的区别在于：第二层nf不翻倍，最后一层只用一个Block
        self.stage_1 = self._make_layer(Block, nf, increase_dim = False, n = layers[0])
        self.stage_2 = self._make_layer(Block, nf, increase_dim = True, n = layers[1])
        self.stage_3 = self._make_layer(Block, 2*nf, increase_dim = True, n = layers[2])
        self.stage_4 = Block(
            4 * nf, increase_dim=False, last_relu=False, downsampling=self._downsampling_type
        )

        if pooling_config["type"] == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif pooling_config["type"] == "weldon":
            self.pool = pooling.WeldonPool2d(**pooling_config)
        else:
            raise ValueError("Unknown pooling type {}.".format(pooling_config["type"]))
        # 输出维度
        self.out_dim = 4 * nf
        # 如果最后一层要再卷一下
        if final_layer in (True, "conv"):
            self.final_layer = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, bias=False)
        # 如果有最后一层，那么在特征提取后加上BN和线性分类层
        elif isinstance(final_layer, dict):
            if final_layer["type"] == "one_layer":
                self.final_layer = nn.Sequential(
                    nn.BatchNorm1d(self.out_dim), nn.ReLU(inplace=True),
                    nn.Linear(self.out_dim, int(self.out_dim * final_layer["reduction_factor"]))
                )
                self.out_dim = int(self.out_dim * final_layer["reduction_factor"])
            elif final_layer["type"] == "two_layers":
                self.final_layer = nn.Sequential(
                    nn.BatchNorm1d(self.out_dim), nn.ReLU(inplace=True),
                    nn.Linear(self.out_dim, self.out_dim), nn.BatchNorm1d(self.out_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.out_dim, int(self.out_dim * final_layer["reduction_factor"]))
                )
                self.out_dim = int(self.out_dim * final_layer["reduction_factor"])
            else:
                raise ValueError("Unknown final layer type {}.".format(final_layer["type"]))
        else:
            self.final_layer = None
        # 参数初始化的方式
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        if zero_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn_b.weight, 0)
    #    self._make_layer(Block, nf, increase_dim=False, n=n)
    # Block是残差块，ResidualBlock()
    # ResidualBlock()的构造函数：def __init__(self, inplanes, increase_dim=False, last_relu=False, downsampling="stride"):
    def _make_layer(self, Block, planes, increase_dim=False, n=None):
        layers = []
        # 如果要增加维度的话，就构造一个增加维度的ResidualBlock，每构造一层，planes都翻一倍
        if increase_dim:
            layers.append(
                Block(
                    planes,
                    increase_dim=True,
                    last_relu=False,
                    downsampling=self._downsampling_type
                )
            )
            planes = 2 * planes
        # 构造不增加维度的ResidualBlock
        for i in range(n):
            layers.append(Block(planes, last_relu=False, downsampling=self._downsampling_type))
        # Stage的构造函数：__init__(self, blocks, block_relu=False):
        return Stage(layers, block_relu=self.last_relu)

    @property
    def last_conv(self):
        return self.stage_4.conv_b

    def forward(self, x):
        # x = self.conv_1_3x3(x)
        x = self.conv1(x)
        x = F.relu(self.bn_1(x), inplace=True)
        # stage_1返回的是Stage，make_layer里面的操作是n个Block
        feats_s1, x = self.stage_1(x)
        feats_s2, x = self.stage_2(x)
        feats_s3, x = self.stage_3(x)
        # stage_4是一个单独的Residual Block
        x = self.stage_4(x)
        # raw_features是把x直接过了pool、flattern和fc之后得到的
        raw_features = self.end_features(x)
        features = self.end_features(F.relu(x, inplace=False))

        if self.all_attentions:
            attentions = [*feats_s1, *feats_s2, *feats_s3, x]
        else:
            attentions = [feats_s1[-1], feats_s2[-1], feats_s3[-1], x]

        return {"raw_features": raw_features, "features": features, "attention": attentions}

    def end_features(self, x):
        x = self.pool(x)
        # 把x reshape
        x = x.view(x.size(0), -1)

        if self.final_layer is not None:
            x = self.final_layer(x)

        return x


def myresnet_gzy(n=5, **kwargs):
    return MyResNetGZY([2,2,2,2], **kwargs)
