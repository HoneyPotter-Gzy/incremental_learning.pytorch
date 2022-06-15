#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time: 2022/6/6 21:21  
@Author: Zheyuan Gu
@File: main.py   
@Description: None
'''

from myUtils.utils import config_parse
from incremental_model_finetune.incremental_finetune_train \
    import IncrementalFinetuneMethod


if __name__=="__main__":
    model_name = "XXX_new_finetune"
    # conf_dict = config_parse(r'../conf.ini', 'incremental_old')
    conf_dict = config_parse(r'../conf.ini', 'incremental_all_test')
    # conf_dict = config_parse(r'../conf.ini', 'incremental_all_test')
    class_num = int(conf_dict['class_num'])
    epoch = int(conf_dict['epoch'])
    batch_size = int(conf_dict['batch_size'])
    learning_rate = float(conf_dict['lr'])
    data_path = conf_dict['source_data_path']
    model_load_path = conf_dict['model_load_path']
    if 'add_class_num' in conf_dict:
        add_class_num = int(conf_dict['add_class_num'])

    IFM = IncrementalFinetuneMethod(class_num, epoch, batch_size, learning_rate,
                                    model_name, model_load_path)

    # def __init__ (self, class_num, epoch, batch_size, learning_rate,
    #               model_name = "RANDOM_MODEL_NAME", model_path = None,
    #               device = "cuda", multi_gpu = True, loss_function = "CE",
    #               optimizer = "SGD"):

    IFM.load_dataset(data_path)
    IFM.increment_class(add_class_num)
    # IFM.train()
    IFM.test()





