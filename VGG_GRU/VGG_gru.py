import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo


class VggNet(nn.Module):    # nn.Module: 모든 신경망 모듈의 기본이 되는 클래스
                             # 각 layer, 함수 등 신경망의 구성요소를 이 클래스 안에서 정의
                             # 매개변수를 encapsulation하는 방법
                             # GPU로 이동, exporting, loading 등의 작업을 위한 helper 제공
                             
	def __init__(self):    # 초기화 함수
        
		super(VggNet, self).__init__()   # super: 자식 클래스에서 부모 클래스의 내용을 사용
		
        
		## VGG
        ## 5개의 conv layer, 1개의 fc layer에 대한 정의  ##
        
        # conv1
		self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)    # 입력 채널, 출력 채널, 필터 크기
		self.relu1_1 = nn.ReLU(inplace=True)
		self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
		self.relu1_2 = nn.ReLU(inplace=True)
		self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # Max pooling over a 2*2 window
                                                                # → 1/2

		# conv2
		self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
		self.relu2_1 = nn.ReLU(inplace=True)
		self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
		self.relu2_2 = nn.ReLU(inplace=True)
		self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # → 1/4

		# conv3
		self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
		self.relu3_1 = nn.ReLU(inplace=True)
		self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
		self.relu3_2 = nn.ReLU(inplace=True)
		self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
		self.relu3_3 = nn.ReLU(inplace=True)
		self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # → 1/8

		# conv4
		self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
		self.relu4_1 = nn.ReLU(inplace=True)
		self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu4_2 = nn.ReLU(inplace=True)
		self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu4_3 = nn.ReLU(inplace=True)
		self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # → 1/16

		# conv5
		self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu5_1 = nn.ReLU(inplace=True)
		self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu5_2 = nn.ReLU(inplace=True)
		self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
		self.relu5_3 = nn.ReLU(inplace=True)

		self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # → 1/32

		# fc6: fully connected layer
		self.fc6 = nn.Linear(512*2*2, 4096)    # torch.nn.Linear(input features, output features, bias=True)
                                               # 지난 output 채널 수 * input dimension (height * width)

		self.gru = nn.GRU(4096,128, batch_first=True)

		self.classify = nn.Linear(128, 3)   # 3 classes - understand, neutral, not understand
		self.dropout = nn.Dropout(p=0.8)    # unit(hidden and visible)을 randomly 무시(drop out)
                                            # → reduce overfitting & improve generalization error
                                            # p=0.5: for retaining 각 node의 the output in a hidden layer
                                            # p=0.8: for retaining inputs from the visible layer

                    

# Forward 함수만 정의하면 변화도를 계산하는 backward 함수는 autograd를 사용하여 자동으로 정의된다..고 한다

	def forward(self, x):

        # conv_ layer1
		h = x            # 아마 h means hidden
		h = self.relu1_1((self.conv1_1(h)))
		h = self.relu1_2((self.conv1_2(h)))
		h = self.pool1(h)

        # conv_ layer2
		h = self.relu2_1((self.conv2_1(h)))
		h = self.relu2_2((self.conv2_2(h)))
		h = self.pool2(h)


        # conv_ layer3
		h = self.relu3_1((self.conv3_1(h)))
		h = self.relu3_2((self.conv3_2(h)))
		h = self.relu3_3((self.conv3_3(h)))
		h = self.pool3(h)

        # conv_ layer4
		h = self.relu4_1((self.conv4_1(h)))
		h = self.relu4_2((self.conv4_2(h)))
		h = self.relu4_3((self.conv4_3(h)))
		h = self.pool4(h)

        # conv_ layer5
		h = self.relu5_1((self.conv5_1(h)))
		h = self.relu5_2((self.conv5_2(h)))
		h = self.relu5_3((self.conv5_3(h)))
		h = self.pool5(h)

        # fc_ layer6
		h = h.view(-1, self.num_flat_features(h))      # view: reshape the tensor
		x = F.relu(self.fc6(h))                        # -1: when row수 모름 but sure of the column수
		x = x.view(1, -1, self.num_flat_features(x))   # flatten this to give it to the fc layer
                                                       # output의 shape과 똑같게  (org value: 1,16,4096)
                                                       # batchsize,sequence_length,data_dim

		# VGG output → GRU
		x, hn = self.gru(x)    

		x = self.dropout(x)
		
		x = torch.mean(x,1)

		x = self.classify(x)

		return x

	def num_flat_features(self, x):
		size = x.size()[1:]   
		num_features = 1
		for s in size:
			num_features *= s
		return num_features