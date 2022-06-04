#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2022/6/4 0:30  
@Author: Zheyuan Gu
@File: t-sne_origin.py   
@Description: None
'''
import pickle
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

device = 'cuda'
batch_size = 32
source_data = r'dataset/12class-train-traffic-data.pkl'

with open(source_data, 'rb') as f:
    train_img_path = pickle.load(f)
    train_img_label = pickle.load(f)

# train_dataset = MyDataset(train_img_path, train_img_label)
# train_loader = DataLoader(train_dataset, batch_size = batch_size,
#                           shuffle = True,
#                           pin_memory = True if device == "cuda" else False)

train_img_label = torch.tensor(train_img_label)  # 转为tensor
_, label = torch.max(train_img_label.data, 1)
label = label.tolist()


print('Begining......') #时间会较长，所有处理完毕后给出finished提示
tsne_2D = manifold.TSNE(n_components=2, init='pca', random_state=0) #调用TSNE
result_2D = tsne_2D.fit_transform(train_img_path)

print('Finished......')
#调用上面的两个函数进行可视化
fig1 = plot_embedding_2D(result_2D, label,'t-SNE')
plt.show(fig1)

