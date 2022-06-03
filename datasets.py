import torch
import cv2
import os
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Compose, ToTensor
import random
import json
# from utils import read_split_data, one_hot

classNum=14


class MyDataset(Dataset):
    def __init__ (self, imgs_path: list, imgs_class: list):
        self.imgs_path = imgs_path
        self.imgs_class = imgs_class
        # self.size = (28, 28)
        # self.transforms=Compose([
        #     Resize(self.size),
        #     ToTensor(),
        # ])
        self.transforms = Compose([
            ToTensor()
        ])

        self.dataAug = None

    def __len__ (self):
        return len(self.imgs_path)

    def __getitem__ (self, item):
        # img = Image.open(self.imgs_path[item])
        img = cv2.imread(self.imgs_path[item])
        # print(img.shape)
        label = self.imgs_class[item]

        img = self.transforms(img)
        # label = torch.tensor(label).reshape(-1, classNum)
        label = torch.tensor(label)
        # label = one_hot(label, classNum).reshape(-1, classNum)
        if self.dataAug is not None:
            img = self.dataAug(img)

        return img, label

def read_split_data (rootpath: str, val_rate: float = 0.2):
    """
    用于读取和分割数据集
    :param rootpath:
    :param val_rate:
    :return:
    """
    random.seed(1)
    assert os.path.exists(rootpath), "dataset root: {} doesn't exist.".format(
        rootpath)

    # TODO: 修改json文件存在性检测，读取相同的类别idx映射
    pcap_class = [classname for classname in os.listdir(rootpath)
                  if os.path.isdir(os.path.join(rootpath, classname))]

    pcap_class.sort()

    class_idx = dict((int(k), int(v)) for k, v in enumerate(pcap_class))
    json_content = json.dumps(dict((v, k) for v, k in class_idx.items()),
                              indent = 4)
    if not os.path.exists(r'data_label_dict2.json'):
        with open(r'data_label_dict2.json', 'w') as json_file:
            json_file.write(json_content)

    train_img_path = []
    train_img_label = []
    val_img_path = []
    val_img_label = []
    each_class_num = []

    for cls in pcap_class:
        cls_path = os.path.join(rootpath, cls)
        imgs_path = [os.path.join(rootpath, cls, i) for i in
                     os.listdir(cls_path)]
        img_class = class_idx[int(cls)]
        each_class_num.append(len(imgs_path))
        val_path = random.sample(imgs_path, k = int(len(imgs_path) * val_rate))

        for img_file in imgs_path:
            if img_file in val_path:
                val_img_path.append(img_file)
                val_img_label.append(img_class)
            else:
                train_img_path.append(img_file)
                train_img_label.append(img_class)

    print("{} images are added to the dataset, ".format(sum(each_class_num)))
    print("in which {} are for training, ".format(len(train_img_path)))
    print("and {} for validation.".format(len(val_img_path)))

    return train_img_path, train_img_label, val_img_path, val_img_label
