import os
import json
import random
import torch
from torch import tensor
import configparser
import matplotlib.pyplot as plt

def one_hot (x, classNum = 14):
    '''
    将label转化为one-hot形式
    :param x:
    :param classNum:
    :return:
    '''
    return torch.eye(classNum)[x, :].float()

def save_model (epochs, model, optimizer, criterion, model_name = 'model'):
    '''
    保存训练模型
    :param epochs:
    :param model:
    :param optimizer:
    :param criterion:
    :return:
    '''
    path = r'../outputs/models_{}'.format(model_name)
    if not os.path.exists(path):
        os.mkdir(path)
    # torch.save("model_e{}.pth".format(e), model.module.parameters())
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, '{}/model_{}e.pth'.format(path, epochs))

def save_plots (train_acc, test_acc, train_loss, test_loss, title):
    '''
    绘制并保存训练/验证集正确率和loss折线图
    :param train_acc:
    :param valid_acc:
    :param train_loss:
    :param valid_loss:
    :return:
    '''
    plt.figure(figsize = (10, 7))
    plt.plot(train_acc, color = 'green', linestyle = '-', label = 'train_acc')
    plt.plot(test_acc, color = 'blue', linestyle = '-', label = 'test_acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../outputs/figs/accuracy_{}.png'.format(title))

    plt.figure(figsize = (10, 7))
    plt.plot(train_loss, color = 'orange', linestyle = '-',
             label = 'train_loss')
    plt.plot(test_loss, color = 'blue', linestyle = '-', label = 'test_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../outputs/figs/loss_{}.png'.format(title))

# save_plots([0.2, 0.7, 0.9], [0.15, 0.62, 0.88], [2.2, 1.1, 0.3], [3.1, 1.6, 0.5], name='model_test')


def cal_acc_for_each_class(test_pred, test_true, total_num_dict, correct_num_dict) ->None:
    '''

    :param test_pred: 模型预测值，单标签数字
    :param test_true: 模型真实值，单标签数字
    :return: dict, 类别数: 正确率
    '''
    test_pred = test_pred.cpu().numpy()
    test_true = test_true.cpu().numpy()

    for i in range(len(test_pred)):
        cls = test_true[i]
        if cls not in correct_num_dict:
            correct_num_dict[cls] = 0
            total_num_dict[cls] = 1
        else:
            total_num_dict[cls] += 1
        if test_pred[i] == test_true[i]:
            correct_num_dict[cls] += 1

    # for k, v in total_num_dict:
    #     acc_num_dict[k] = acc_num_dict[k] / total_num_dict[k]

def config_parse(cfg_path, args):
    cfg_path = r"../conf.ini"

    conf = configparser.ConfigParser()
    conf.read(cfg_path, encoding = "utf-8")
    base_items = conf.items('base_args')
    spec_items = conf.items(args)  # 返回一个list，里面的内容是元组

    config_dict = {}
    for item in base_items:
        config_dict[item[0]] = item[1]

    for item in spec_items:
        config_dict[item[0]] = item[1]

    return config_dict

