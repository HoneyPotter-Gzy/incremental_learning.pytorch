#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2022/6/4 18:14  
@Author: Zheyuan Gu
@File: model.py   
@Description: None
'''
import torch
import torch.nn as nn

from torchvision.models.resnet import resnet18
from base_model.model import MyResnet, get_MyResnet
from typing import Type, Any, Callable, Union, List, Optional
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


class IncrementalResnet(MyResnet):
    def __init__(self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        ResNet.__init__(self, block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)

    def increment_classes(self, n):
        '''
        :param n: 新增加的类别数
        :return: None
        增加n个新类别，将最后一个fc层（输出维度为旧类别数m）替换为新fc（输出维度为当前总类别数m+n），并同步之前的fc到新fc上
        '''
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        self.fc=nn.Linear(in_features, out_features+n, bias = False)
        self.fc.weight.data[:out_features] = weight
        self.classNum += n


def get_MyIncrementalResnet() -> IncrementalResnet:
    model = IncrementalResnet(BasicBlock, [2, 2, 2, 2])
    return model