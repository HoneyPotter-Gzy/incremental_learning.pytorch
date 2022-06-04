#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2022/5/26 19:36
@Author: Zheyuan Gu
@File: model.py.py   
@Description: None
'''

import torch
import torch.nn as nn

from torchvision.models.resnet import resnet18
from typing import Type, Any, Callable, Union, List, Optional
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
# from utils import read_split_data, one_hot

classNum=12
device=torch.device("cuda")

class MyResnet(ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(MyResnet, self).__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=3, bias=False)
        self.maxpool = nn.Identity()
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1,
                                       dilate=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilate=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=False)
        self.fc = nn.Linear(512, classNum)
        self.classificator = nn.Softmax(-1)

    def forward(self, x):
        # x=self.conv1(x)
        # x=self.maxpool(x)
        # x=self.layer1(x)
        # x=self.layer2(x)
        # x=self.layer3(x)
        # x=self.layer4(x)
        # feature = x
        # x=self.fc(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feature = x
        x = self.fc(x)
        return feature, x


def get_MyResnet() -> ResNet:
    model = MyResnet(BasicBlock, [2, 2, 2, 2])
    return model


# class MyModel(nn.Module):
#     def __init__(self, classnum=12, dim=28 * 28):
#         super(MyModel, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.BatchNorm1d(dim),
#             nn.ReLU(),
#             nn.Linear(dim, int(dim ** 0.5)),
#             nn.BatchNorm1d(int(dim ** 0.5)),
#             nn.ReLU(),
#             nn.Linear(int(dim ** 0.5), 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, classnum),
#             nn.Softmax(-1)
#         )
#
#     def forward(self, input):
#         B, C, H, W = input.shape
#         input = input.mean(dim=1).reshape(B, -1)
#         return self.model(input)


# class MyModel(nn.Module):  # 完整的model，包括resnet18、fc和softmax，输出预测概率
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.feature_extractor=resnet18(pretrained = True)
#         self.feature_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         in_channel = self.feature_extractor.fc.in_features
#         self.feature_extractor.fc=nn.Linear(in_channel, classNum)
#         self.classificator = nn.Sequential(
#             nn.Softmax(-1)
#         )
#         # in_channel = model.fc.in_features
#         # model.fc = torch.nn.Linear(in_channel, classNum)
#         # return model.to(device)
#
#     def forward(self, x):
#         y=self.feature_extractor(x)
#         y=self.classificator(y)
#         return y


# def build_model(pretrained=True, fine_tune=True, num_class=classNum):
#     if pretrained:
#         print('[INFO]: Loading pre-trained weights...')
#     else:
#         print('[INFO]: Not loading pre-trained weights...')
#
#     model = MyModel()
#     print(model)
#     return model

    # if fine_tune:
    #     print('[INFO]: Fine-tuning all layers...')
    #     for params in model.parameters():
    #         params.requires_grad=True
    #
    # else:
    #     print('[INFO]: Freezing hidden layers...')
    #     for params in model.parameters():
    #         params.requires_grad=False

    # in_channel = model.fc.in_features
    # model.fc = torch.nn.Linear(in_channel, classNum)
    # return model.to(device)
# model=MyModel()
# print(model)
