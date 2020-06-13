import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo


class BasicBlock1(nn.Module):

    def __init__(self, planes):
        super(BasicBlock1, self).__init__()

        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1)
        # parameters initialization
        nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv2.bias, 0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out += residual
        return out


""" 
class BasicBlock1(nn.Sequential):
    def __init__(
        self, planes,  bias=False, bn=True, act=nn.ReLU(True)):

        super(BasicBlock1, self).__init__()

        m = []
        for i in range(2):
            m.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(planes))
            if act is not None:
                m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        out = res + x
        return out
 """

#model = ResLSTMNet(BasicBlock1, [1, 2, 5, 3])

class ResLSTMNet(nn.Module):

    def __init__(self, block, layers):
        super(ResLSTMNet, self).__init__()

        self.conv1_a = nn.Conv2d(1,  32, 3, stride=1)
        self.bn1_a = nn.BatchNorm2d(32)
        self.conv1_b = nn.Conv2d(32, 64, 3, stride=1)
        self.bn1_b = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.conv2   = nn.Conv2d(64,  128, 3, stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.conv3   = nn.Conv2d(128, 256, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.conv4   = nn.Conv2d(256, 512, 3, stride=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.layer4 = self._make_layer(block, 512, layers[3])

        self.fc5_new = nn.Linear(512*2*2, 512)
        
        #final classification
        self.fc8_final = nn.Linear(128, 3)

        # parameters initialization #
        layers = [self.conv1_a,self.conv1_b,self.conv2,self.conv3,self.conv4]

        self.gru1 = nn.GRU(512, 128, batch_first=True)
        self.gru2 = nn.GRU(512, 128, batch_first=True)
        self.gru3 = nn.GRU(512, 128, batch_first=True)

        self.dropCon  = nn.Dropout(p=0.4)
        self.dropCon1 = nn.Dropout(p=0.4)
        
        self.dropout1 = nn.Dropout(p=0.8)
        self.dropout2 = nn.Dropout(p=0.8)
        self.dropout3 = nn.Dropout(p=0.8)

        self.fc5_pool1b = nn.Linear(64*30*30,512)
        self.fc5_pool3  = nn.Linear(256*6*6,512)
        

        self.InitParam(layers)


    def InitParam(self,layers):
        for layer in layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

#self.layer4 = self._make_layer(block, 512, 3)
    def _make_layer(self, block, planes, blocks):

        layers = []
        for i in range(0, blocks):
          layers.append(block(planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        # print('x.shape : ',x.shape)
        x = F.relu(self.bn1_a(self.conv1_a(x)))
        x = self.dropCon(x)
        x_pool1b = F.max_pool2d(F.relu(self.bn1_b(self.conv1_b(x))),2, stride=2)

        x = self.layer1(x_pool1b)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))),2, stride=2)

        x = self.layer2(x)
        x = self.dropCon1(x)
        x_pool3 = F.max_pool2d(F.relu(self.bn3(self.conv3(x))),2, stride=2)

        x = self.layer3(x_pool3)
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))),2, stride=2)
        x = self.layer4(x)
        # print('layer4(x).shape : ',x.shape)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc5_new(x)
        
        # print('x_pool1b.shape : ',x_pool1b.shape)
        x_pool1b = x_pool1b.view(-1, self.num_flat_features(x_pool1b))
        x_pool1b = self.fc5_pool1b(x_pool1b)

        # print('x_pool3.shape : ', x_pool3.shape)
        x_pool3 = x_pool3.view(-1, self.num_flat_features(x_pool3))
        x_pool3 = self.fc5_pool3(x_pool3)

        # print('x.shape : ', x.shape)
        x = x.unsqueeze(0)
        # print('x.shape: ',x.shape)
        x, hx1 = self.gru1(x)
        # print('x.shape, hx1.shape: ',x.shape ,hx1.shape)
        x = self.dropout1(x)

        x_pool1b = x_pool1b.unsqueeze(0)
        # print('x_pool1b.shape: ',x_pool1b.shape)
        x_pool1b, hx2 = self.gru2(x_pool1b)
        # print('x_pool1b.shape, hx1.shape: ',x_pool1b.shape ,hx2.shape) 
        x_pool1b = self.dropout2(x_pool1b)

        x_pool3 = x_pool3.unsqueeze(0)
        # print('x_pool3.shape: ',x_pool3.shape)
        x_pool3, hx3 = self.gru3(x_pool3)
        # print('x_pool3.shape, hx3.shape: ',x_pool3.shape ,hx3.shape) 
        x_pool3 = self.dropout3(x_pool3)

        x_fusion = x+ x_pool1b+ x_pool3

        # print('f_fusion.shape : ',x_fusion.shape)
        # print('f_fusion : ', x_fusion)
        x = torch.mean(x_fusion,1)
        x = self.fc8_final(x)
        # print('fc8.final(x).shape : ',x.shape)
        # print('fc8.final(x) : ', x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]   
        num_features = 1
        for s in size:
            num_features *= s
        # print('num_flat_features : ',num_features)
        return num_features
