import torch
from torch.utils import data
import glob
import os
import cv2
import random
import numpy as np

class FACEDATA(data.Dataset):
    def __init__(self, args, train = False):
        self.args = args
        self.train = train
        self.img_size = args.img_size
        self.data_list = glob.glob(os.path.join(args.dataset_dir, '**', '*.jpg'), recursive = True)
        self.len = len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        aug_img = self.transform(img)
        label = int(img_path.split('_')[-1].split('.')[0])  # 0,1,2
        return aug_img, label

    def __len__(self):
        return self.len


    def transform(self, img):
        ndim = img.ndim
        # print('data_loader img.shape : ',img.shape)
        if ndim==2:   #grayscale image
            h,w = img.shape
            # img = np.expand_dims(img, axis=0)  #(1,2)
            img = img.reshape(h,w)
            stacked_img = np.stack((img,) * 3, axis=0)
        # else :
        #     h,w,c=img.shape
        #     if c==3:
        #         #color to gray
        #         pass

        aug_img = self.augment(stacked_img)
        aug_img = torch.from_numpy(aug_img).float()  #numpy array to tensor
        # print('data_loader aug_img.shape : ',aug_img.shape)

        return aug_img


    def augment(self, img, hflip=True, rot=True): #c,h,w
        hflip = hflip and random.random()<0.5
        vflip = rot and random.random() <0.5
        # rot90 = rot and random.random() <0.5

        if hflip: img=img[:,:,::-1].copy()    #::-1 --> 역순으로
        if vflip: img=img[:,::-1,:].copy()
        # if rot90: img=img.transpose(0,2,1).copy()  #img.transpose(0,2,1)--> 1,2의 위치 바꿈

        return img

"""
    #for real-time
    def get_real_data(self):
        img_shape=(1,self.img_size, self.img_size)
        crop_img=self.face_detection(self)
        #resize
        resize_img=np.resize(crop_img, img_shape)
        aug_img = self.augment(resize_img)
        return aug_img

    #for real-time
    def face_detection(self):
        #face detect
        #to gray scale
        pass

"""
def get_train_valid_dataloader(args):
    dataset = FACEDATA(args)
    train_len = int(len(dataset)*args.train_ratio)
    valid_len = len(dataset)-train_len
    train_dataset, valid_dataset = data.random_split(dataset, lengths=[train_len, valid_len])

    print('Num of train dataset : {}, Num of valid dataset : {}'.format(len(train_dataset), len(valid_dataset)))
    train_dataloader = data.DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    valid_dataloader = data.DataLoader(dataset=valid_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)

    return train_dataloader, valid_dataloader

