import cv2
import numpy as np   
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from VGG_gru import VggNet
import time
import argparse


def predect_emotion(opt, model, data):  #c, h, w
    global p_time
    global iter_Num

    start_time = time.time()

    with torch.no_grad():
        
        data = torch.from_numpy(data).float().to(opt.device)
        # data = data.to('cpu')

        output = model(data)
        softmax = nn.Softmax(1)
        pred = softmax(output)
        pred = pred.squeeze()
        #print(pred.shape)
        idx = pred.data.max(0)[1] 
        # print('pred : {}, idx : {}'.format(pred,idx))
        p_time += time.time()-start_time
        # print('p_time: ',p_time)
        # print('d_time: ',d_time)
        print('Avg prediction time : {:.3f} sec'.format(p_time/iter_Num))

    return pred,idx


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Facial Expression real-time test')
    parser.add_argument('--device', default='cpu', choices = ['cpu', 'cuda'])
    opt = parser.parse_args()

    ## 모델 만들기
    model = VggNet()

    # Face detection XML load and trained model loading
    face_detector = cv2.CascadeClassifier('files/haarcascade_frontalface_default.xml')
    emotion_classifier = torch.load('../../files/our_model.pth',map_location=torch.device('cpu'))
    #face_detector = cv2.CascadeClassifier('face_detection/haarcascade_frontalface_alt.xml')
    #emotion_classifier = torch.load('../../data/face_data/checkpoint/2020-05-05_models_epoch_0451_loss_0.008156_acc_0.943.pth',map_location=torch.device('cpu'))
    model.load_state_dict(emotion_classifier['model'])

    if torch.cuda.is_available() and opt.device == 'cuda':
        print('setting GPU')
        opt.device = 'cuda'
        model = model.to(opt.device)
    else : 
        opt.device = 'cpu'

    EMOTIONS = ["Not Understand","Neutral","Understand"]

    # Video capture using webcam
    camera = cv2.VideoCapture(0)
    images = np.zeros((4,1,64,64))

    iter_Num = 0
    d_time = 0.0
    p_time = 0.0

    while True:
        s4 = time.time()
        for i in range(4):
            while True:
                # Capture image from camera
                ret, frame = camera.read()  # ret: frame 사용 가능 여부
                                            # frame: 캡쳐된 image arr vector
                # to gray scale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray,
                                                    scaleFactor=1.1,
                                                    minNeighbors=5,
                                                    minSize=(30,30))

                # Perform emotion recognition only when face is detected
                if len(faces)>0:
                    break
            #detect face
            face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = face
            # Resize the image to 48x48 for neural network
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            #cv2.imwrite('D:/Jupyter_Project/ex.png',roi)
            roi = roi.astype("float") / 255.0 

            images[i,0,:,:] = roi
            
        d_time += time.time()-s4
        iter_Num += 1
        print("[*]iter ",iter_Num)
        print('Avg detection time: {:.3f} sec'.format(d_time/iter_Num))
        preds, label = predect_emotion(opt, model,images)        

        # Assign labeling
        cv2.putText(frame, EMOTIONS[label], (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

        # Create empty image
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        # Label printing
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)    
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 255, 0), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Open two windows
        ## Display image ("Emotion Recognition")
        ## Display probabilities of emotion
        cv2.imshow('Emotion Recognition', frame)
        cv2.imshow("Probabilities", canvas)
        
        # q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clear program and close windows
    camera.release()
    cv2.destroyAllWindows()