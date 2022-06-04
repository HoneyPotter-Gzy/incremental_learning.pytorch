import os
import json
import random
import torch
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
    path = r'outputs/models_{}'.format(model_name)
    if not os.path.exists(path):
        os.mkdir(path)
    # torch.save("model_e{}.pth".format(e), model.module.parameters())
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, '{}/model_{}e.pth'.format(path, epochs))

def save_plots (train_acc, test_acc, train_loss, test_loss,
                model_name = 'model'):
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
    plt.savefig('outputs/figs/accuracy_{}.png'.format(model_name))

    plt.figure(figsize = (10, 7))
    plt.plot(train_loss, color = 'orange', linestyle = '-',
             label = 'train_loss')
    plt.plot(test_loss, color = 'blue', linestyle = '-', label = 'test_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/figs/loss_{}.png'.format(model_name))

# save_plots([0.2, 0.7, 0.9], [0.15, 0.62, 0.88], [2.2, 1.1, 0.3], [3.1, 1.6, 0.5], name='model_test')
