#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
import datetime
import cv2
from torch.utils.data import Dataset
from PIL import Image
# from utils import *
from torchvision import transforms
from glob import glob



class FER(Dataset):

    def __init__(self, opt):
        # opt: mode('train', 'valid'), img_size, train_dir

        self.opt = opt
        self.mode = opt.mode
        self.img_size = opt.img_size
        self.img_list = glob(os.path.join(opt.train_dir), '*.jpg')
        self.len = len(self.img_list)

    # 클래스에서의 __len__ & __getitem 구현 (double underbar method="special method")
    # → len()이용 가능 / index 접근으로 원하는 값을 얻을 수 있음
    # random module의 choice, sort 이용 가능
    # slicing (ex- carddeck[:3] ), in (ex- card('Q','heart') in deck >> True) 이용 가능
    def __len__(self):  
        return self.nSamples

    def __getitem__(self, index):   

        if self.mode == 'train' or self.mode=='valid':
            img_path = self.img_list[index]
            img = cv2.imread(img_path)      
            ### tensor로 바꿔줘야 하나..? 만약 그렇다면 아래 코드 주석 해제
            # img   = torch.from_numpy(img).float()         
            # or, img = torch.tensor(img)
            aug_img = self.transform(img)

            label = int(img_path.split('_')[-1].split('.')[0]) #0-not understand ,1-neutral ,2-understand
                                            # -1: 뒤에서 첫번째 / ← split 순서 ←
            return aug_img, label
        else:
            img = self.get_real_data()      # for real-time
            return img

    def transform(self, img):
        ndim = img.ndim
        if ndim == 2:
            h,w = img.shape
            img = img.reshpae(1,h,w)
        else :
            h,w,c = img.shape
            if c == 3:
                # color to gray
                img = np.dot(img[...,:3],[0.2999, 0.587, 0.144])
                # or, img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                pass
        aug_img = self.augment(img)
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
        img_shape = (1,self.img_size, self.img_size)    ### 이렇게 똑같이 써줘도 똑똑하게 h, w 구분하나..?
        crop_img = self.face_detection(self)
        #resize
        resize_img = np.resize(crop_img, img_shape)
        aug_img = self.augment(resize_img)
        return aug_img

    #For real-time
    def face_detection(self, dir):
        # dir = [face_detector_dir, emot_classifier_dir, save_dir]


        #face detect
        camera = cv2.VideoCapture(0)
        # 얼굴 detect할 classifier (ex - lbpcascade_frontalface_default.xml')
        face_detector = cv2. CascadeClassifier(dir[0])
        # emotion classifier 모델 (ex - 'emotion_model.hdf5')
        emotion_classifier = load_model(dir[1], compile=False)
        classes = [0, 1, 2] # ['not understand','neutral','understand']
        while True:
            # Capture image from camera
            ret, frame = camera.read()  # ret: frame 사용 가능 여부
                                        # frame: 캡쳐된 image arr vector
            # to gray scale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Face detection in frame
            faces = face_detector.detectMultiScale(gray,
                                                    scaleFactor=1.1,
                                                    minNeighbors=5,
                                                    minSize=(30,30))

            # Perform emotion recognition only when face is detected
            if len(faces) > 0:
                # For the largest image
                face = sorted(faces, reverse=True, key=lambda x: (x[2]-x[0]) * (x[3]-x[1]))[0]
                # reverse=True: 내림차순 / lamda: x(faces)를 받아서 ':'이후의 식 반환
                (fX, fY, fW, fH) = face
                # Resize the image for neural network
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (self.img_size, self.img_size))   ### size 이렇게 써도 되나..
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                crop_img = np.expand_dims(roi, axis=0)   # 0: 제일 앞에 있는 차원 추가 

                # Emotion predict
                preds = emotion_classifier.predict(crop_img)[0]
                label = classes[preds.argmax()]

                # Save files

                fourcc = cv2.VideoWriter_fourcc(*'XVID')    # ('코덱') → 인코딩 방식 설정
                record = False
                now = datetime.datetime.now().strftime("%d_%H-%M-%S")   # 이름으로 쓸 현재 시간

                key = cv2.waitKey(10)   # 10ms마다 누른 키보드 값 갱신
                if key == 26:   # <ctrl + z> to capture
                    print("캡쳐")
                    cv2.imwrite(dir[2] + str(now) + ".jpg", crop_img)
                elif key == 24:   # <ctrl + x> to record
                    print("녹화 시작")
                    record = True
                    video = cv2.VideoWriter(dir[2] + str(now) + ".avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                                                    # (path, codec, fps, width & heigth) 
                                                    ### 얘는 크기를 어떻게 할까?
                elif key == 3:   # <ctrl + c> to stop
                    print("녹화 중지")
                    record = False
                    video.release()

                if record == True:
                    print("녹화 중..")
                    video.write(frame)

            # q to quit
            if key == ord('q'):
                print("카메라 종료")
                break

        # Clear program and close windows
        camera.release()
        cv2.destroyAllWindows()

        return crop_img
        


def get_train_valid_dataloader(opt):
    dataset = FER(opt)
    train_len = int(len(dataset)*opt.train_ratio)
    valid_len = len(dataset)-train_len
    train_dataset, valid_dataset = data.random_split(dataset, lengths=[train_len, valid_len])

    train_dataloader = data.DataLoader(dataset=train_dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=2)
    valid_dataloader = data.DataLoader(dataset=valid_dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=2)
    
    return train_dataloader, valid_dataloader




