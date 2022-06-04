#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2022/6/4 18:51  
@Author: Zheyuan Gu
@File: train.py.py   
@Description: None
'''

import torch
from torch.utils.data import DataLoader
import pickle
from dataset_process.dataset_new import MyDataset
from utils.utils import save_model, save_plots
# from base_model.model import get_MyResnet
from incremental_model_finetune.model import get_MyIncrementalResnet

model_name = 'incremental_resnet_modified'
# source_data = r'dataset/12class-train-traffic-data.pkl'
old_data = r'D:\毕设\preprocess\code_new\dataset\first_class_[0, 1, 2, 3, 4, 5, 6, 7].pkl'
new_data = r'D:\毕设\preprocess\code_new\dataset\second_class_[8, 9, 10, 11].pkl'
classNum = 12
batch_size = 16
epoch = 200
lr = 1e-2
weight_decay = 1e-5
# lbs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
old_classes = [0, 1, 2, 3, 4, 5, 6, 7]
new_classes = [8, 9, 10, 11]


def train (device = torch.device("cuda"), multi_gpu = False,
           batch_size = batch_size,
           epoch = epoch, lr = lr, weight_decay = weight_decay,
           load_ckpt = None):

    print('[INFO]: Computation device: {}'.format(device))

    with open(old_data, 'rb') as f:
        train_img_path = pickle.load(f)
        train_img_label = pickle.load(f)
        test_img_path = pickle.load(f)
        test_img_label = pickle.load(f)

    # 加载数据
    train_dataset = MyDataset(train_img_path, train_img_label)
    train_loader = DataLoader(train_dataset, batch_size = batch_size,
                              shuffle = True,
                              pin_memory = True if device == "cuda" else False)

    test_dataset = MyDataset(test_img_path, test_img_label)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False,
                             pin_memory = True if device == "cuda" else False)

    model = get_MyIncrementalResnet()
    model.to(device = device)

    if load_ckpt:
        model.load_state_dict(torch.load(load_ckpt))
    if multi_gpu:
        model = torch.nn.DataParallel(model)

    lossfunc = torch.nn.CrossEntropyLoss(reduction = "mean")
    optimizer = torch.optim.SGD(model.parameters(), lr = lr,
                                weight_decay = weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                     [epoch * 0.5])

    best_pfm = 0.0

    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    cumu_train_loss = 0.0
    cumu_train_acc = 0.0
    cumu_test_loss = 0.0
    cumu_test_acc = 0.0

    for e in range(epoch):
        model.train()
        train_loss = 0.0
        train_correct_num = 0
        train_cnt = 0

        for img, label in train_loader:  # 对训练集中每一个batch
            train_cnt += 1
            img = img.to(device = device)
            label = label.to(device = device, dtype = torch.long)
            y = model(img)

            _, train_preds = torch.max(y.data, 1)
            _, train_true = torch.max(label.data, 1)

            loss = lossfunc(y, train_true).sum()  # 一个batch的loss之和
            train_loss += loss.item()

            train_correct_num += (train_preds == train_true).sum().item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_train_loss = train_loss / train_cnt
        epoch_train_acc = 100. * (train_correct_num) / len(
            train_loader.dataset)
        print("epoch: {} train, loss: {}, acc: {}".format(e, epoch_train_loss,
                                                          epoch_train_acc))

        model.eval()

        test_loss = 0.0
        test_correct_num = 0
        test_cnt = 0

        for img, label in test_loader:
            test_cnt += 1
            img = img.to(device = device)
            label = label.to(device = device, dtype = torch.long)

            with torch.no_grad():
                y = model(img)

            _, test_preds = torch.max(y.data, 1)
            _, test_true = torch.max(label.data, 1)

            loss = lossfunc(y, test_true).sum()
            test_loss += loss.item()

            test_correct_num += (test_preds == test_true).sum().item()

        epoch_test_loss = test_loss / test_cnt
        epoch_test_acc = 100. * (test_correct_num) / len(test_loader.dataset)
        print("epoch: {} test, loss: {}, acc: {}".format(e, epoch_test_loss,
                                                         epoch_test_acc))

        print("*********************************************")

        if (e + 1) % 20 == 0:
            train_loss_list.append(cumu_train_loss / 20)
            train_acc_list.append(cumu_train_acc / 20)
            test_loss_list.append(cumu_test_loss / 20)
            test_acc_list.append(cumu_test_acc / 20)

            cumu_train_loss = 0.0
            cumu_train_acc = 0.0
            cumu_test_loss = 0.0
            cumu_test_acc = 0.0
        else:
            cumu_train_loss += epoch_train_loss
            cumu_train_acc += epoch_train_acc
            cumu_test_loss += epoch_test_loss
            cumu_test_acc += epoch_test_acc

        if epoch_test_acc > best_pfm:
            best_pfm = epoch_test_acc
            save_model(e, model, optimizer, lossfunc, model_name = model_name)

    save_plots(train_acc_list, test_acc_list, train_loss_list, test_loss_list,
               name = model_name)

train()
