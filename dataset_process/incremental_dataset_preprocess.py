#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2022/6/3 15:57  
@Author: Zheyuan Gu
@File: incremental_dataset_preprocess.py   
@Description: 划分数据集，将其中的8类和另4类分开保存成pickle，进行增量学习的finetune
'''

import os
import numpy as np
import pickle

def save_pickle (train_img, train_label, test_img, test_label, filename):
    # train_img = np.concatenate(train_img, axis = 0)
    # train_label = np.concatenate(train_label, axis = 0)
    # test_img = np.concatenate(test_img, axis = 0)
    # test_label = np.concatenate(test_label, axis = 0)
    train_img = np.array(train_img)
    train_label = np.array(train_label)
    test_img = np.array(test_img)
    test_label = np.array(test_label)

    # 转为one-hot
    classNum=8
    train_label = np.eye(classNum)[train_label]
    test_label = np.eye(classNum)[test_label]
    with open(filename, 'wb') as f:
        pickle.dump(train_img, f)
        pickle.dump(train_label, f)
        pickle.dump(test_img, f)
        pickle.dump(test_label, f)

source_data = r'../dataset/12class-train-traffic-data.pkl'

# 通过修改这里的class来控制哪些类别是增量类别
first_class = [0, 1, 2, 3, 4, 5, 6, 7]
second_class = [8, 9, 10, 11]

# 都是ndarray
with open(source_data, 'rb') as f:
    train_img_path = pickle.load(f)
    train_img_label = pickle.load(f)
    test_img_path = pickle.load(f)
    test_img_label = pickle.load(f)

train_img_label = np.argmax(train_img_label, axis = 1)
test_img_label = np.argmax(test_img_label, axis = 1)

first_train_samples = []
second_train_samples = []
first_train_labels = []
second_train_labels = []

first_test_samples = []
second_test_samples = []
first_test_labels = []
second_test_labels = []

for i in range(train_img_path.shape[0]):
    if train_img_label[i] in first_class:
        first_train_samples.append(train_img_path[i])
        first_train_labels.append(train_img_label[i])
    elif train_img_label[i] in second_class:
        second_train_samples.append(train_img_path[i])
        second_train_labels.append(train_img_label[i])
    else:
        raise ValueError("Invalid class")

for i in range(test_img_path.shape[0]):
    if test_img_label[i] in first_class:
        first_test_samples.append(test_img_path[i])
        first_test_labels.append(test_img_label[i])
    elif test_img_label[i] in second_class:
        second_test_samples.append(test_img_path[i])
        second_test_labels.append(test_img_label[i])
    else:
        raise ValueError("Invalid class")

first_class_filename = r'first_class_{}.pkl'.format(first_class)
second_class_filename = r'second_class_{}.pkl'.format(second_class)
save_pickle(first_train_samples, first_train_labels, first_test_samples,
            first_test_labels, first_class_filename)
save_pickle(second_train_samples, second_train_labels, second_test_samples,
            second_test_labels, second_class_filename)
