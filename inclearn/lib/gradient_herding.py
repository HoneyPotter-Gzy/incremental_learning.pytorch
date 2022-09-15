#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2022/9/15 10:20  
@Author: Zheyuan Gu
@File: gradient_herding.py   
@Description: None
'''
import torch
import torch.nn.functional as F

class HerdingSelection:
    def __init__(self):
        self.inner_class_score = dict()
        self.inner_task_score = dict()
        self.cross_task_score = dict()
        self.total_score = dict()

    def inner_class_score_calculation(self, class_representation, label):
        pass

    def inner_task_score_calculation(self, curr_task_grad):
        task_grads = curr_task_grad[0]
        x_grads = task_grads[1]
        mean_grad = torch.mean(x_grads, dim = 0)

        for i in range(len(x_grads)):
            x_grad = x_grads[i]
            self.inner_task_score[i] = F.cosine_similarity(x_grad, mean_grad,
                                                      dim = 0)

        self.inner_task_score = sorted(self.inner_task_score.items(),
                                  key = lambda x: x[1].item(), reverse = True)

    def cross_task_score_calculation(self, curr_task_grad, old_task_grad):
        cross_task_score = dict()

        task_grads = curr_task_grad[0]
        x_grads = task_grads[1]

        old_grads = old_task_grad[0]
        old_x_grads = old_grads[1]
        mean_old_grad = torch.mean(old_x_grads, dim = 0)

        for i in range(len(x_grads)):
            x_grad = x_grads[i]
            cross_task_score[i] = F.cosine_similarity(x_grad, mean_old_grad,
                                                      dim = 0)
        cross_task_score = sorted(cross_task_score.items(),
                                  key = lambda x: x[1].item(), reverse = True)

    def minibatch_selection(self, n_samples, class_representation, curr_task_grad, old_task_grad = {}, task_idx = 0):  # 进行两次动态选择
        '''
        :param n_samples:
        :param curr_task_grad:
        :param old_task_grad:
        :param class_representation:
        :param task_idx: 任务序号，默认是0，即第一个任务，非0时表示需要考虑旧任务
        :return:
        '''

        self.inner_class_score = self.inner_class_score_calculation()
        self.inner_task_score = self.inner_task_score_calculation()
        assert len(self.inner_class_score) == len(self.inner_task_score)

        self.total_score = dict()

        if task_idx:
            self.cross_task_score = self.cross_task_score_calculation()
            assert len(self.inner_class_score) == len(self.cross_task_score)

            for i in range(len(self.inner_class_score)):
                self.total_score[i] = self.inner_class_score[i] + \
                                      self.inner_task_score[i] + \
                                      self.cross_task_score[i]

        else:
            for i in range(len(self.inner_task_score)):
                self.total_score[i] = self.inner_class_score[i] + \
                                      self.inner_task_score[i]

        self.total_score[i] = sorted(self.total_score,
                                     key = lambda x: x[1].item(), reverse = True)