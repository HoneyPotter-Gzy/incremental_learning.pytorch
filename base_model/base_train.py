#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2022/6/6 15:28  
@Author: Zheyuan Gu
@File: base_train.py   
@Description: None
'''

import torch
from torch.utils.data import DataLoader
import pickle
# from datasets import read_split_data, MyDataset
from dataset_process.dataset_new import MyDataset
from myUtils.utils import save_model, save_plots, cal_acc_for_each_class
# from model import MyModel
from base_model.model import get_MyResnet

class BaseMethod():
    '''
    基方法类，对所有数据做全量训练。
    load_dataset(): 加载data_loader
    train(): 训练
    test(): 仅测试
    '''

    def __init__ (self, class_num, epoch, batch_size, learning_rate,
                  model_name = "RANDOM_MODEL_NAME", model_path = None,
                  device = "cuda", multi_gpu = True, loss_function = "CE",
                  optimizer = "SGD"):
        self.class_num = class_num
        self.model = get_MyResnet(class_num)
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.multi_gpu = multi_gpu
        self.softmax = torch.nn.Softmax(-1)
        if loss_function == "CE":
            self.loss_function = torch.nn.CrossEntropyLoss(reduction = "mean")
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr = self.learning_rate,
                                             weight_decay = 1e-5)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor = 0.1, verbose = True)
        self.train_loader = None
        self.test_loader = None

        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(device = self.device)

    def load_dataset (self, source_data):
        with open(source_data, 'rb') as f:
            train_img_path = pickle.load(f)
            train_img_label = pickle.load(f)
            test_img_path = pickle.load(f)
            test_img_label = pickle.load(f)
        train_dataset = MyDataset(train_img_path, train_img_label)
        self.train_loader = DataLoader(train_dataset,
                                       batch_size = self.batch_size,
                                       shuffle = True,
                                       pin_memory = True if self.device == "cuda" else False)

        test_dataset = MyDataset(test_img_path, test_img_label)
        self.test_loader = DataLoader(test_dataset, batch_size = 1,
                                      shuffle = False,
                                      pin_memory = True if self.device == "cuda" else False)

    def train (self):
        # 用于绘制loss变化曲线
        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []
        cumu_train_loss = 0.0
        cumu_train_acc = 0.0
        cumu_test_loss = 0.0
        cumu_test_acc = 0.0

        best_pfm = 0.0

        for e in range(self.epoch):

            self.model.train()
            train_loss = 0.0
            train_correct_num = 0
            train_cnt = 0
            for img, label in self.train_loader:
                train_cnt += 1
                img = img.to(device = self.device)
                label = label.to(device = self.device, dtype = torch.long)

                y = self.model(img)
                # y=self.softmax(y)

                _, train_preds = torch.max(y.data, 1)
                _, train_true = torch.max(label.data, 1)

                # 计算loss，无需softmax
                loss = self.loss_function(y, train_true).sum()
                train_loss += loss
                train_correct_num += (train_preds == train_true).sum().item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_train_loss = train_loss / train_cnt
            epoch_train_acc = 100. * train_correct_num / len(
                self.train_loader.dataset)
            print("epoch: {} train, loss: {}, acc: {}".format(e,
                                                              epoch_train_loss,
                                                              epoch_train_acc))

            epoch_test_loss, epoch_test_acc = self.__test(e)

            if (e + 1) % 10 == 0:
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
                save_model(e, self.model, self.optimizer, self.loss_function,
                           model_name = self.model_name)

            self.lr_scheduler.step(epoch_test_loss)

    def __test (self, e):
        self.model.eval()

        test_loss = 0.0
        test_correct_num = 0
        test_cnt = 0

        for img, label in self.test_loader:
            test_cnt += 1
            img = img.to(device = self.device)
            label = label.to(device = self.device, dtype = torch.long)

            with torch.no_grad():
                y = self.model(img)

            _, test_preds = torch.max(y.data, 1)
            _, test_true = torch.max(label.data, 1)

            loss = self.loss_function(y, test_true).sum()
            test_loss += loss.item()
            test_correct_num += (test_preds == test_true).sum().item()

        epoch_test_loss = test_loss / test_cnt
        epoch_test_acc = 100. * (test_correct_num) / len(
            self.test_loader.dataset)
        print("epoch: {} test, loss: {}, acc: {}".format(e, epoch_test_loss,
                                                         epoch_test_acc))

        print("*********************************************")
        return epoch_test_loss, epoch_test_acc

    def test (self):
        if self.model_path == None:
            raise ValueError("Invalid Loading Model Path")

        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)

        state = torch.load(self.model_path)
        self.model.load_state_dict(state['model_state_dict'])

        self.model.to(device = self.device)
        self.model.eval()

        test_correct_num = 0
        test_cnt = 0

        total_num_dict = { }
        correct_num_dict = { }

        for img, label in self.test_loader:
            test_cnt += 1
            img = img.to(device = self.device)
            label = label.to(device = self.device, dtype = torch.long)

            with torch.no_grad():
                y = self.model(img)

            _, test_preds = torch.max(y.data, 1)
            _, test_true = torch.max(label.data, 1)

            test_correct_num += (test_preds == test_true).sum().item()

            cal_acc_for_each_class(test_preds, test_true, total_num_dict,
                                   correct_num_dict)

        epoch_test_acc = 100. * (test_correct_num) / len(
            self.test_loader.dataset)
        print("Total acc: {}".format(epoch_test_acc))

        acc_num_dict = { }
        for k in sorted(total_num_dict.keys()):
            acc_num_dict[k] = correct_num_dict[k] / total_num_dict[k]
            print("class: {}, total: {}, "
                  "correct: {}, acc: {}".format(k, total_num_dict[k],
                                                correct_num_dict[k],
                                                acc_num_dict[k]))

        print("*********************************************")
