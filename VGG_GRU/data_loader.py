import os
import cv2
import glob
import torch
import random
import datetime
import numpy as np
from utils import *
from PIL import Image
from torchvision import transforms
from torch.utils import data

class FER(data.Dataset):

    def __init__(self, opt, mode):
        # opt: mode('train', 'valid'), img_size, train_dir

        self.opt = opt
        self.mode = mode
        self.img_size = opt.img_size
        self.length = opt.length    # batch for kinda trick
        self.iter = opt.iter
        # self.img_list = glob(os.path.join(opt.train_dir), '*.jpg')
        
        if self.mode == 'train':
            img_list_0 = glob.glob(os.path.join(opt.train_dir, 'Not_Understand', '*.jpg'))
            img_list_1 = glob.glob(os.path.join(opt.train_dir, 'Neutral', '*.jpg'))
            img_list_2 = glob.glob(os.path.join(opt.train_dir, 'Understand', '*.jpg'))
        elif self.mode == 'valid':
            img_list_0 = glob.glob(os.path.join(opt.valid_dir, 'Not_Understand', '*.jpg'))
            img_list_1 = glob.glob(os.path.join(opt.valid_dir, 'Neutral', '*.jpg'))
            img_list_2 = glob.glob(os.path.join(opt.valid_dir, 'Understand', '*.jpg'))
        self.img_list_0 = sorted(img_list_0)
        self.img_list_1 = sorted(img_list_1)
        self.img_list_2 = sorted(img_list_2)
        # 클래스 별 sample 개수
        self.len0 = len(self.img_list_0)
        self.len1 = len(self.img_list_1)
        self.len2 = len(self.img_list_2)
        print('Number of each class images >> len0 : {}, len1 : {}, len2 : {}'.format(self.len0, self.len1, self.len2))


    # 클래스에서의 __len__ & __getitem 구현 (double underbar method="special method")
    # → len()이용 가능 / index 접근으로 원하는 값을 얻을 수 있음
    # random module의 choice, sort 이용 가능
    # slicing (ex- carddeck[:3] ), in (ex- card('Q','heart') in deck >> True) 이용 가능
    def __len__(self):  
        '''return self.nSamples'''
        if self.mode == 'train':
            if self.iter:
                return self.iter
            else : 
                return int(((self.len0 + self.len1 + self.len2)))
        elif self.mode == 'valid': #valid는 iter에 상관없이 항상 모든 데이터 보게끔
            return int(((self.len0 + self.len1 + self.len2)))

    def __getitem__(self, index):   
        r = np.random.randint(9)
        img_path = []
        seq = np.zeros((self.length, 1, self.img_size, self.img_size))

        if self.mode == 'train' or self.mode=='valid':
            if (r%3) == 0: img_list = self.img_list_0; num = self.len0
            elif (r%3) == 1: img_list = self.img_list_1; num = self.len1
            else : img_list = self.img_list_2; num = self.len2

            idx = random.sample(range(num), self.length)
            for i, img_num in enumerate(idx) : 
                img_path.append(img_list[img_num])
                img = cv2.imread(img_list[img_num], cv2.IMREAD_GRAYSCALE)
                aug_img = self.transform(img)
                # print('aug_img.shape :',aug_img.shape)
                seq[i, :, :, :] = aug_img

            seq = torch.from_numpy(seq).float()
            label=int(img_path[0].split('_')[-1].split('.')[0]) #0-not understand ,1-neutral ,2-understand
            label = torch.LongTensor([label])
            # print('FER/ img_path : {}, label : {}'.format(img_path[0].split('\\')[-1], label))
            return seq, label

        else :
            img = self.get_real_data()
            return img       

    def transform(self, img):
        ndim = img.ndim
        if ndim == 2:
            img = np.expand_dims(img, axis=0)   # 0: 제일 앞에 차원 추가 
            # stacked_img = np.stack((img,) * 3, axis=0)    0: first dim 

        else :
            h,w,c = img.shape
            if c == 3:
                # color to gray
                pass
        aug_img = self.augment(img)
        # aug_img = torch.from_numpy(aug_img).float() # np arr to tensor
        # print('data_loader aug_img.shape : ',aug_img.shape)
        return aug_img


    def augment(self, img, hflip=True, vflip=True, rot=True): # c,h,w
        hflip = hflip and random.random()<0.5   # random module의 random 함수
        vflip = vflip and random.random() <0.5    
        rot90 = rot and random.random() <0.5

        if hflip: img = img[:,:,::-1].copy()    # arr[::-1] 처음부터 끝까지 역순으로
        if vflip: img = img[:,::-1,:].copy()
        if rot90: img = img.transpose(0,2,1).copy()     # idx 순서 (c, w, h)
       
        # rotate 다양한 각도 ver.
        # rotated_img = np.array(img.rotate(45))# width, height

        return img

    #for real-time
    def get_real_data(self):
        img_shape = (1,self.img_size, self.img_size)    
        crop_img = self.face_detection(self)
        #resize
        resize_img = np.resize(crop_img, img_shape)
        aug_img = self.augment(resize_img)
        return aug_img


def get_dataloader(opt,mode):

    dataset = FER(opt, mode)
    length = len(dataset)

    print('Length of {} dataloader : {}'.format(opt.mode, length))
    if mode == 'train':
        dataloader = data.DataLoader(dataset=dataset,
                                batch_size=1,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=opt.num_workers)
    elif mode == 'valid':
        dataloader = data.DataLoader(dataset=dataset,
                                batch_size=opt.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=opt.num_workers)
    
    return dataloader