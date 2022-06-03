#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2022/6/3 19:33  
@Author: Zheyuan Gu
@File: save_figs.py   
@Description: None
'''

import os
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Compose, ToTensor
import random
import json
import numpy as np
import pickle
import torch
from torchvision.models.resnet import resnet18
from torchvision.utils import save_image

source_data = r'dataset/12class-train-traffic-data.pkl'
root_path=r'D:\毕设\preprocess\code_new\pngs\test'
with open(source_data, 'rb') as f:
    x_train = pickle.load(f)
    y_train = pickle.load(f)
    x_test = pickle.load(f)
    y_test = pickle.load(f)
#
# x_train=torch.tensor(x_train)
# y_train=torch.tensor(y_train)
# x_test=torch.tensor(x_test)
# y_test=torch.tensor(y_test)
# print(x_train.shape)
# print(y_train.shape)
#
x_test = torch.tensor(x_test).reshape(4647, 1, 28, 28)
y_test = torch.max(torch.tensor(y_test).data, 1)[1].tolist()
# print(x_train)
# print(y_train)

num=36822
for idx in range(num):
    pic_path=os.path.join(root_path, str(y_test[idx]))
    print(pic_path)
    pic_name=os.path.join(pic_path, str(idx)+".png")
    print(pic_name)

    save_image(x_test[idx].data, pic_name)





