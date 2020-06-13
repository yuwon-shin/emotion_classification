import cv2
import numpy as np   
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from numpy import asarray
from torchvision import models
from Resnet_GRU.resnet_gru import BasicBlock1, ResLSTMNet
from face_detection.MTCNN.detector import FaceDetector
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

def convert_to_square(bboxes):
    """Convert bounding boxes to a square form.
    Arguments:
        bboxes: a float numpy array of shape [n, 5].
    Returns:
        a float numpy array of shape [n, 5],
            squared bounding boxes.
    """

    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w * 0.5 - max_side * 0.5
    square_bboxes[:, 1] = y1 + h * 0.5 - max_side * 0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Facial Expression real-time test')
    parser.add_argument('--device', default='cpu', choices = ['cpu', 'cuda'])
    parser.add_argument('--real', action='store_true')
    opt = parser.parse_args()

    ## 모델 만들기
    model = ResLSTMNet(BasicBlock1, [1, 2, 5, 3])

    # Face detection and trained model loading
    detector = FaceDetector(opt)
    emotion_classifier = torch.load('Resnet_GRU/rsenet_gru_best_checkpoint_0.943.pth',map_location=torch.device(opt.device))
    #emotion_classifier = torch.load('../../data/face_data/checkpoint/2020-05-05_models_epoch_0451_loss_0.008156_acc_0.943.pth',map_location=torch.device('cpu'))


    model.load_state_dict(emotion_classifier['model'])

    if torch.cuda.is_available() and opt.device == 'cuda':
        print('setting GPU')
        opt.device = 'cuda'
        model = model.to(opt.device)
    else : 
        print('using CPU')
        opt.device = 'cpu'

    EMOTIONS = ["Not Understand","Neutral","Understand"]

    # Video capture using webcam
    camera = cv2.VideoCapture(0) if opt.real else cv2.VideoCapture('../../data/face_data/test/test.mp4')
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
                # to RGB
                rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # DETECT!
                box, landmark = detector.detect(rgb)
                if len(box)>0:
                    break

            # Resize the image to 64x64 for neural network
            img_list = []
            size = 64
            square_bboxes = convert_to_square(box)
            for b in square_bboxes:
                face_img = rgb.crop((b[0], b[1], b[2], b[3]))
                face_img = face_img.resize((size, size), Image.BILINEAR)
                img_list.append(face_img)
            roi = asarray(img_list[0]) 
            
            # to gray
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            cv2.imwrite('eunhye.png', gray)
            gray = gray.astype("float") / 255.0 


            images[i,0,:,:] = gray
     
            
        d_time += time.time()-s4
        iter_Num += 1
        print("[*]iter ",iter_Num)
        print('Avg detection time: {:.3f} sec'.format(d_time/iter_Num))
        preds, label = predect_emotion(opt, model, images)        

        # Assign labeling
        # cv2.putText(frame, EMOTIONS[label], (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # Create empty image
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        # Label printing
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)    
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 255, 0), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        img_marked = rgb.copy()
        draw = ImageDraw.Draw(img_marked)

        for b in box:
            draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline="white")

        for p in landmark:
            for i in range(5):
                draw.ellipse(
                    [(p[i] - 1.0, p[i + 5] - 1.0), (p[i] + 1.0, p[i + 5] + 1.0)],
                    outline="blue",
                )
        
        frame = cv2.cvtColor(asarray(img_marked), cv2.COLOR_RGB2BGR)

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