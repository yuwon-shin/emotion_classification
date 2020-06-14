import cv2
import numpy as np   
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from Resnet_GRU.resnet_gru import BasicBlock1, ResLSTMNet
import time
import argparse
from face_detection.BlazeFace.blazeface import BlazeFace


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
        pred = pred.squeeze().to('cpu')
        #print(pred.shape)
        idx = pred.data.max(0)[1] 
        # print('pred : {}, idx : {}'.format(pred,idx))
        p_time += time.time()-start_time
        # print('p_time: ',p_time)
        # print('d_time: ',d_time)
        pred = pred.numpy()
        print('Avg prediction time : {:.3f} sec'.format(p_time/iter_Num))

    return pred,idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial Expression real-time test with Blazeface detection')
    parser.add_argument('--device', default='cpu', choices = ['cpu', 'cuda'])
    parser.add_argument('--prototype', type=int, default = 1, choices=[1,2,3])
    opt = parser.parse_args()

    ## 모델 만들기
    model = ResLSTMNet(BasicBlock1, [1, 2, 5, 3])

    # Face detection load and trained model loading
    face_detector = BlazeFace(opt)
    face_detector.load_weights('face_detection/BlazeFace/blazeface.pth')
    face_detector.load_anchors('face_detection/BlazeFace/anchors.npy')

    # emotion_classifier = torch.load('../../data/face_data/checkpoint/2020-05-05_models_epoch_0451_loss_0.008156_acc_0.943.pth',map_location=torch.device(opt.device))
    emotion_classifier = torch.load('Resnet_GRU/rsenet_gru_best_checkpoint_0.943.pth',map_location=torch.device(opt.device))
    model.load_state_dict(emotion_classifier['model'])

    if torch.cuda.is_available() and opt.device == 'cuda':
        print('setting GPU')
        opt.device = 'cuda'
        model = model.to(opt.device)
        face_detector = face_detector.to(opt.device)
    else : 
        print('using CPU')
        opt.device = 'cpu'

    EMOTIONS = ["Not Understand","Neutral","Understand"]

    # Video capture using webcam
    camera = cv2.VideoCapture('../../data/face_data/test/prototype{}.mp4'.format(opt.prototype))
    # images = np.zeros((4, 3, 480, 640))
    detections = []
    faces = np.zeros((6,4,1,64,64)) #사람별
    grids = np.zeros((6,128,128,3)) #한 프레임에서

    iter_Num = 0
    d_time = 0.0
    p_time = 0.0

    while True:
        s4 = time.time()
        ymins = []
        xmins = []
        ymaxs = []
        xmaxs = []
        for i in range(4):
            while True : 
                # Capture image from camera
                ret, frame = camera.read()  # ret: frame 사용 가능 여부
                                            # frame: 캡쳐된 image arr vector
                # to gray scale
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                grid1 = rgb[:240,:284,:]
                grid2 = rgb[:240,284:568,:]
                grid3 = rgb[:240,568:852,:]
                grid4 = rgb[240:480,:284,:]
                grid5 = rgb[240:480,284:568,:]
                grid6 = rgb[240:480,568:852,:]

                grids[0,:,:,:] = cv2.resize(grid1,(128,128))
                grids[1,:,:,:] = cv2.resize(grid2,(128,128))
                grids[2,:,:,:] = cv2.resize(grid3,(128,128))
                grids[3,:,:,:] = cv2.resize(grid4,(128,128))
                grids[4,:,:,:] = cv2.resize(grid5,(128,128))
                grids[5,:,:,:] = cv2.resize(grid6,(128,128))
                
                d = face_detector.predict_on_batch(grids)                

                if len(d) > 0:
                    # detections.append(d)
                    break

            #d = d.cpu().numpy()
            #print(len(d))
            #if d.ndim == 1:
            #    d = np.expand_dims(d,axis=0)
            for p,dd in enumerate(d):
                dd = dd.cpu().numpy()
                ymin = int(dd[0,0]*240) # 세로
                xmin = int(dd[0,1]*284) # 가로
                ymax = int(dd[0,2]*240)
                xmax = int(dd[0,3]*284)
                new_ymin = int(ymin - 0.3*(ymax-ymin))
                ymin = new_ymin if new_ymin > 0 else 0

                ymin = p//3*240+ymin
                ymax = p//3*240+ymax
                xmin = p%3*284+xmin
                xmax = p%3*284+xmax

                ymins.append(ymin)
                xmins.append(xmin)
                ymaxs.append(ymax)
                xmaxs.append(xmax)

                roi = rgb[ymin:ymax, xmin:xmax, :]
                cv2.imwrite('origin_{}.png'.format(i),roi)
        
                roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                roi = cv2.resize(roi, (64,64))
                roi = roi.astype('float')/255.0
                faces[p,i,0,:,:] = roi
 
        d_time += time.time()-s4
        iter_Num += 1
        print("[*]iter ",iter_Num)
        print('Avg detection time: {:.3f} sec'.format(d_time/iter_Num))
        
        preds_sum = np.zeros(3)
        for pp in range(6):
            preds, label = predect_emotion(opt, model,faces[pp])
            preds_sum += preds        
            # Assign labeling
            cv2.putText(frame, EMOTIONS[label], (xmins[pp], ymins[pp] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
            cv2.rectangle(frame, (xmins[pp], ymins[pp]), (xmaxs[pp], ymaxs[pp]), (255, 0, 0), 2)

        # Create empty image
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        # Label printing
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds_sum/6)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)    
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (255, 0, 0), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Open two windows
        ## Display image ("Emotion Recognition")
        ## Display probabilities of emotion
        cv2.imshow('Emotion Recognition', frame)
        cv2.imshow("Avg Probabilities", canvas)
        
        # q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clear program and close windows
    camera.release()
    cv2.destroyAllWindows()
