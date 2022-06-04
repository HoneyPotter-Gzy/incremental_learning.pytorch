#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2022/5/26 19:39  
@Author: Zheyuan Gu
@File: train.py.py   
@Description: None
'''


import torch
from torch.utils.data import DataLoader
import pickle
# from datasets import read_split_data, MyDataset
from dataset_process.dataset_new import MyDataset
from utils.utils import save_model, save_plots
# from model import MyModel
from base_model.model import get_MyResnet

# from utils import read_split_data, one_hot

# trainpath = r'../4_Png/Train'
# testpath = r'../4_Png/Test'
model_name='resnet_modified'
source_data = r'dataset/12class-train-traffic-data.pkl'
classNum = 12
batch_size = 16
epoch = 200
lr = 1e-2
weight_decay = 1e-5
lbs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def train (device = torch.device("cuda"), multi_gpu = False,
           batch_size = batch_size,
           epoch = epoch, lr = lr, weight_decay = weight_decay,
           load_ckpt = None):
    print('[INFO]: Computation device: {}'.format(device))

    # train_img_path, train_img_label, val_img_path, val_img_label = \
    #     read_split_data(trainpath)  # 获取训练集和验证集路径+标签
    # test_img_path, test_img_label, a, b = read_split_data(testpath,
    #                                                       val_rate = 0)

    with open(source_data, 'rb') as f:
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


    # train_dataset = MyDataset(imgs_path = train_img_path,
    #                           imgs_class = train_img_label)
    # train_loader = DataLoader(train_dataset, batch_size = batch_size,
    #                           shuffle = True,
    #                           pin_memory = True if device == "cuda" else False)

    # val_dataset = MyDataset(imgs_path = val_img_path,
    #                         imgs_class = val_img_label)
    # val_loader = DataLoader(val_dataset, batch_size = batch_size,
    #                         shuffle = False,
    #                         pin_memory = True if device == "cuda" else False)

    # test_dataset = MyDataset(imgs_path = test_img_path,
    #                          imgs_class = test_img_label)
    # test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False,
    #                          pin_memory = True if device == "cuda" else False)

    # model = MyModel()
    model = get_MyResnet()
    model.to(device = device)

    if load_ckpt:
        model.load_state_dict(torch.load(load_ckpt))
    if multi_gpu:
        model = torch.nn.DataParallel(model)

    lossfunc = torch.nn.CrossEntropyLoss(reduction = "mean")
    optimizer = torch.optim.SGD(model.parameters(), lr = lr,
                                weight_decay = weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        [epoch * 0.5])

    best_pfm = 0.0
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    test_loss_list = []
    test_acc_list = []
    cumu_train_loss = 0.0
    cumu_train_acc = 0.0
    cumu_val_loss = 0.0
    cumu_val_acc = 0.0
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
            # print(img.shape)
            # print(label.shape)
            # print(label.shape)

            # y = model(img).to(device=device)
            y = model(img)
            # print(y.shape)

            # print(y)
            # print(label)
            _, train_preds = torch.max(y.data, 1)
            _, train_true = torch.max(label.data, 1)

            # loss = lossfunc(y, label).sum()  # 一个batch的loss之和
            loss = lossfunc(y, train_true).sum()  # 一个batch的loss之和
            train_loss += loss.item()

            # _, train_preds = torch.max(y.data, 1)
            # _, train_true = torch.max(label.data, 1)
            # train_correct_num += (train_preds == label).sum().item()
            train_correct_num += (train_preds == train_true).sum().item()


            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # print(model.parameters().grad)
        # for name, parms in model.named_parameters():
        #     print(parms.grad)

        # lr_scheduler.step()
        epoch_train_loss = train_loss / train_cnt
        epoch_train_acc = 100. * (train_correct_num) / len(
            train_loader.dataset)
        print("epoch: {} train, loss: {}, acc: {}".format(e, epoch_train_loss,
                                                        epoch_train_acc))

        # # 开始val集
        # model.eval()
        # # y_true = []
        # # y_score = []
        # val_loss = 0.0
        # val_correct_num = 0
        # val_cnt = 0
        #
        # for img, label in val_loader:
        #     val_cnt += 1
        #     img = img.to(device = device)
        #     # label = label.to(device = device)
        #     label = label.to(device = device, dtype = torch.long)
        #
        #     with torch.no_grad():  # ?????????
        #         y = model(img)
        #
        #     loss = lossfunc(y, label).sum()
        #     val_loss += loss.item()
        #     _, val_preds = torch.max(y.data, 1)
        #     # _, val_true = torch.max(label.data, 1)
        #     val_correct_num += (val_preds == label).sum().item()
        #     # y_score.append(y.detach().cpu().numpy()[:, -1])
        #     # y_true.append(label.detach().cpu().numpy())
        #
        # # y_true = np.concatenate(y_true, 0)
        # # y_score = np.concatenate(y_score, 0)
        # # y_score = y_score
        # # _, preds = torch.max(y.data, 1)
        # # _, true = torch.max(label.data, 1)
        # # epoch_correct_num+=(preds==true).sum().item()
        # # auc = metrics.roc_auc_score(y_true, y_score)
        # # fpr, tpr, thr = metrics.roc_curve(y_true, y_score)
        # epoch_val_loss = val_loss / val_cnt
        # epoch_val_acc = 100. * (val_correct_num) / len(val_loader.dataset)
        # print("epoch: {} val, loss: {}, acc: {}".format(e, epoch_val_loss,
        #                                                 epoch_val_acc))
        # # print("epoch: {} train, loss: {}, fpr: {}, tpr: {}"
        # #       .format(e, l_sum / sum, fpr, tpr))

        model.eval()
        # y_true = []
        # y_score = []
        test_loss = 0.0
        test_correct_num = 0
        test_cnt = 0

        for img, label in test_loader:
            test_cnt += 1
            img = img.to(device = device)
            # label=label.to(device=device)
            label = label.to(device = device, dtype = torch.long)

            with torch.no_grad():
                y = model(img)

            _, test_preds = torch.max(y.data, 1)
            _, test_true = torch.max(label.data, 1)

            # loss = lossfunc(y, label).sum()
            loss = lossfunc(y, test_true).sum()
            test_loss += loss.item()

            test_correct_num += (test_preds == test_true).sum().item()
            # y_score.append(y.detach().cpu().numpy()[:, -1])
            # y_true.append(label.detach().cpu().numpy())

        # y_true = np.concatenate(y_true, 0)
        # y_score = np.concatenate(y_score, 0)

        epoch_test_loss = test_loss / test_cnt
        epoch_test_acc = 100. * (test_correct_num) / len(test_loader.dataset)
        print("epoch: {} test, loss: {}, acc: {}".format(e, epoch_test_loss,
                                                         epoch_test_acc))

        print("*********************************************")

        # y_score = y_score
        # # auc = metrics.roc_auc_score(y_true, y_score)
        # fpr, tpr, thr = metrics.roc_curve(y_true, y_score)
        # print("epoch: {} train, loss: {}, fpr: {}, tpr: {}"
        #       .format(e, l_sum / sum, fpr, tpr))

        # print("epoch: {}".format(e), "loss: {}".format(epoch_loss),
        #       "precision: {}".format(epoch_acc))
        if (e + 1) % 20 == 0:
            train_loss_list.append(cumu_train_loss / 20)
            train_acc_list.append(cumu_train_acc / 20)
            # val_loss_list.append(cumu_val_loss / 20)
            # val_acc_list.append(cumu_val_acc / 20)
            test_loss_list.append(cumu_test_loss / 20)
            test_acc_list.append(cumu_test_acc / 20)
            cumu_train_loss = 0.0
            cumu_train_acc = 0.0
            # cumu_val_loss = 0.0
            # cumu_val_acc = 0.0
            cumu_test_loss = 0.0
            cumu_test_acc = 0.0
        else:
            cumu_train_loss += epoch_train_loss
            cumu_train_acc += epoch_train_acc
            # cumu_val_loss += epoch_val_loss
            # cumu_val_acc += epoch_val_acc
            cumu_test_loss += epoch_test_loss
            cumu_test_acc += epoch_test_acc

        if epoch_test_acc > best_pfm:
            best_pfm = epoch_test_acc
            save_model(e, model, optimizer, lossfunc, model_name = model_name)

    save_plots(train_acc_list, test_acc_list, train_loss_list, test_loss_list, name=model_name)
    # torch.save("model_e{}.pth".format(e), model.module.parameters())
    # save_model(epoch, model, optimizer, lossfunc)

train()
# print(torch.cuda.is_available())
