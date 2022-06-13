#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2022/6/13 0:05  
@Author: Zheyuan Gu
@File: myresnet_gzy.py   
@Description: None
'''
import torch
import torch.nn as nn

from typing import Type, Any, Callable, Union, List, Optional
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


class MyResnetGZY(ResNet):
    # TODO: 6.13起来写这部分的网络结构和方法，争取上午成功跑起来icarl！下午研究自己的方法
    def __init__ (
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(MyResnetGZY, self).__init__(block, layers, num_classes,
                                       zero_init_residual, groups,
                                       width_per_group,
                                       replace_stride_with_dilation,
                                       norm_layer)
        self.classNum = num_classes
        self.conv1 = nn.Conv2d(1, 64, kernel_size = (7, 7), stride = (1, 1),
                               padding = 3, bias = False)
        self.maxpool = nn.Identity()
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 1,
                                       dilate = False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 1,
                                       dilate = False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 1,
                                       dilate = False)
        self.fc = nn.Linear(512, self.classNum)
        self.out_dim = 512


def myresnet_gzy(num_classes = 12):
    return MyResnetGZY(BasicBlock, [2, 2, 2, 2], num_classes)