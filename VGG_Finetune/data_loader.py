import torch
from torch.utils import data
from glob import glob
import os
import cv2


class FER(data.Dataset) :

    def __init__(self, data_path, image_size=64, mode='train') :
        self.img_list = glob(os.path.join(data_path, mode, '*.jpg'))
        self.len = len(self.img_list)
        self.mode = mode
        self.image_size = image_size


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_path = self.img_list[index]
        data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        data = torch.tensor(data)

        filename = image_path.split(os.sep)[-1]
        label = int(filename.split('_')[-1].split('.')[0])
        label = torch.tensor(label)

        return data, label


