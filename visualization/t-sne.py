#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2022/6/2 18:01  
@Author: Zheyuan Gu
@File: t-sne.py   
@Description: None
'''

import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from sklearn import manifold
from dataset_process.dataset_new import MyDataset
from torch.utils.data import DataLoader
from base_model.model import get_MyResnet
import os
import torch.nn as nn
from collections import OrderedDict

# class IntermediateLayerGetter(nn.ModuleDict):
#     def __init__(self, model, return_layers):
#         if not set(return_layers).issubset([name for name, _ in model.named_children()]):
#             raise ValueError("return_layers are not present in model")
#
#         orig_return_layers = return_layers
#         return_layers = {k: v for k, v in return_layers.items()}
#         layers = OrderedDict()
#         for name, module in model.named_children():
#             layers[name] = module
#             if name in return_layers:
#                 del return_layers[name]
#             if not return_layers:
#                 break
#
#         super(IntermediateLayerGetter, self).__init__(layers)
#         self.return_layers = orig_return_layers
#
#     def forward(self, x):
#         out = OrderedDict()
#         for name, module in self.named_children():
#             x = module(x)
#             if name in self.return_layers:
#                 out_name = self.return_layers[name]
#                 out[out_name] = x
#         return out

def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str('o'),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    #     plt.
    # temp={}
    # for i in range(data.shape[0]):
    #     # color = plt.cm.Set3(label[i])  # 颜色
    #     lb = label[i]
    #     if lb not in temp:
    #         temp[lb] = []
    #     temp[lb].append(data)
    # for color, datas in temp.items():
    #     plt.scatter(datas[:0], datas[:, 1], color=plt.cm.Set3(color))
    # for i in range(data.shape[0]):
    #     plt.scatter(data[i, 0], data[i, 1], color=plt.cm.Set3(label[i]))
    # plt.scatter(data, label, )
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

device = 'cuda'
batch_size = 512
source_data = r'../dataset/12class-train-traffic-data.pkl'

with open(source_data, 'rb') as f:
    train_img_path = pickle.load(f)
    train_img_label = pickle.load(f)

# 加载数据
train_dataset = MyDataset(train_img_path, train_img_label)
train_loader = DataLoader(train_dataset, batch_size = batch_size,
                          shuffle = True,
                          pin_memory = True if device == "cuda" else False)

model = get_MyResnet().to(device)

if device=='cuda':
    model=torch.nn.DataParallel(model)
# 数据加载
with open(source_data, 'rb') as f:
    a = pickle.load(f)
    b = pickle.load(f)
    x_train = pickle.load(f)
    y_train = pickle.load(f)

model_path=r'E:\gzy\preprocess1\outputs\models_resnet_modified\model_147e.pth'
state = torch.load(model_path)
model.load_state_dict(state['model_state_dict'])

model.eval()
# print(model)
x_f = []
y_pred = []
# return_layers = {'avgpool': 'x_feature'}
# backbone = IntermediateLayerGetter(model.module, return_layers)
# backbone.eval()
print("get features and labels...")
for img, label in train_loader:
    img = img.to(device = device)
    label = label.to(device = device)
    _, label = torch.max(label.data, 1)  # 转换成单数字标签

    with torch.no_grad():
        # out = backbone(img)  # 提取的特征
        # _x = out['x_feature']
        _x, _y = model(img)  # 预测标签，tensor one-hot

    _, _y = torch.max(_y.data, 1)  # 获取特征,tensor 数字

    x_f.append(_x.detach().cpu().numpy())  # 获取特征
    y_pred.append(_y.detach().cpu().numpy())  # 获取预测标签,ndarray
    # 需要ndarray作为tsne的输入

x_f = np.concatenate(x_f, axis = 0)
y_pred = np.concatenate(y_pred, axis = 0)
print("get features and labels done")



# x_train=np.resize(x_train, (36822, 784))
# x_f = np.resize(x_f, (36822, 512))
# y_train = torch.tensor(y_train)
# _, y_train = torch.max(y_train.data, 1)
y_pred = y_pred.tolist()

print("tsne start")
embedding_files = r'x_embedding_test.pkl'
if not os.path.exists(embedding_files):
    tsne_f = manifold.TSNE(n_components=2, init='pca', random_state=501, learning_rate = 20, perplexity= 50)
    x_embedding = tsne_f.fit_transform(x_train)
    with open(embedding_files, 'wb') as f:
        pickle.dump(x_embedding, f)
else:
    with open(embedding_files, 'rb') as f:
        x_embedding=pickle.load(f)

print('tsne finish')
#调用上面的两个函数进行可视化
fig1 = plot_embedding_2D(x_embedding, y_pred,'t-SNE')
plt.savefig("test.png")
plt.show()

# x_embedding=x_train
# x_eMax = x_embedding.max(axis=0)
# x_eMin = x_embedding.min(axis=0)
# x_norm = (x_embedding-x_eMin) / (x_eMax - x_eMin)
#
# plt.figure(figsize=(8,8))
# # print(x_embedding)
# for i in range(x_norm.shape[0]):
#     plt.text(x_norm[i, 0], x_norm[i, 1], str(y_train[i]), color=plt.cm.Set1(y_train[i]), fontdict = {'weight': 'bold', 'size': 9})
# plt.xticks([])
# plt.yticks([])
# plt.show()
# print("wait")

