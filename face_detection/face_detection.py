#LBP Cascade Classifier
#https://github.com/informramiz/Face-Detection-OpenCV


""" 
LBP is a texture descriptor and face is composed of micro texture patterns. 
So LBP features are extracted to form a feature vector to classify a face from a non-face. Following are the basic steps of LBP Cascade classifier algorithm:

1. LBP Labelling: 
    A label as a string of binary numbers is assigned to each pixel of an image.
2. Feature Vector: 
    Image is divided into sub-regions and for each sub-region, a histogram of labels is constructed. 
    Then, a feature vector is formed by concatenating the sub-regions histograms into a large histogram.
3. AdaBoost Learning: 
    Strong classifier is constructed using gentle AdaBoost to remove redundant information from feature vector.
4. Cascade of Classifier: 
    The cascades of classifiers are formed from the features obtained by the gentle AdaBoost algorithm. 
    Sub-regions of the image is evaluated starting from simpler classifier to strong classifier. 
    If on any stage classifier fails, that region will be discarded from further iterations. 
    Only the facial region will pass all the stages of the classifier.

 """

import numpy as np
import cv2
#import matplotlib.pyplot as plt
import time 
import glob
import os
import numpy as np


def detect_faces(f_cascade, colored_img, scaleFactor = 1.01):
    img_copy = np.copy(colored_img)
    #convert the test image to gray image as opencv face detector expects gray images
    #gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(img_copy, scaleFactor=scaleFactor, minNeighbors=5)
    
    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        print(x,y,w,h)
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255), 2)
        
    return img_copy



if __name__ == "__main__":
    #load cascade classifier training file for lbpcascade
    #face_cascade = cv2.CascadeClassifier('./face_detection/lbpcascade_frontalface.xml')
    face_cascade = cv2.CascadeClassifier('./face_detection/haarcascade_frontalface_alt.xml')

    #load test iamge
    fer2013 = '../../data/face_data/fer2013/*.jpg'
    jaffe = '../../data/face_data/jaffe/*.jpg'

    fer2013_list = glob.glob(fer2013)
    jaffe_list = glob.glob(jaffe)
    #fer2013_list = [os.path.join(fer2013, x) for x in os.path.listdir(fer2013)]
    #jaffe_list = [os.path.join(jaffe, x) for x in os.listdir(jaffe)]
    
    #test1 = cv2.imread(fer2013_list[0])
    #test2 = cv2.imread(jaffe_list[0])

    sc = 1.01

    for x in fer2013_list :
        img = cv2.imread(x, cv2.IMREAD_GRAYSCALE) #h,w
        print(img.shape)
        img = cv2.resize(img, (128,128))
        detected_img = detect_faces(face_cascade, img, scaleFactor=sc)
        cv2.imwrite('../../data/face_data/result/%s'%('out-'+str(sc)+'-' + os.path.basename(x)), detected_img)

    for x in jaffe_list :
        img = cv2.imread(x, cv2.IMREAD_GRAYSCALE) #h,w
        print(img.shape)
        detected_img = detect_faces(face_cascade, img, scaleFactor=sc)
        cv2.imwrite('../../data/face_data/result/%s'%('out-'+str(sc)+'-' + os.path.basename(x)), detected_img)

    pass

""" 
#display the gray image using OpenCV
cv2.imshow('Test Imag', gray_img)
cv2.waitKey(100)
cv2.destroyAllWindows() """

