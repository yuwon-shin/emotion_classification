from __future__ import print_function  # for compatibility
import os
import time
import numpy
import random
# import dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import models
from utils import *  # *: utils 안의 모든 이름에 access
from VGG_gru import VggNet
# from tensorboardX import SummaryWriter
import argparse
import subprocess  # 다른 언어로 만들어진 프로그램을 통합 ex) for 오래된 os.*
from tqdm import tqdm
from data_loader import FER, get_dataloader



def valid(opt, model, valid_loader, metric):
    start_time = time.time()

    metric.reset()

    model.eval()

    print('[*]Validation...')

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(tqdm(valid_loader)):

            Batch, T, C, H, W = data.size()

            data = data.squeeze(0)
            data = Variable(data).to(opt.device)

            label = Variable(label.long()).to(opt.device)
            label = label.squeeze(1)

            output = []
            for batch_index in range(Batch):

                output_feature = model(data[batch_index])
                output.append(output_feature)

            output = torch.cat(output, 0)
            pred = output.data.cpu().max(1)[1]

            metric(output, label)
            accuracy, eval_loss = metric.value()
            avg_loss = eval_loss / ((batch_idx + 1) * opt.batch_size)

            print('Validation [{}/{}] accuracy : {:.2f}, loss : {:.6f}'.format(batch_idx, len(valid_loader), accuracy,
                                                                               avg_loss))
            print('acc & loss:', accuracy, '&', eval_loss)
            print('prediction : ', pred)

    print('[*]Validation...')

    return accuracy, avg_loss



if __name__ == '__main__':

    data_dir = '../../data/face_data'

    checkpoint_dir = os.path.join(data_dir, 'checkpoint')
    test_dir = os.path.join(data_dir, 'val')

    parser = argparse.ArgumentParser(description='PyTorch Facial Expression')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for valid/train = 1')
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--length', type=int, default=4,
                        help='data shape : (b, <<l>>, c, h, w) | meaning batch in training | for making each batch containing same class')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--optim', type=str, default='sgd', choices=['adam', 'sgd'])
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--data_dir', type=str, default=data_dir,
                        help='dataset path')
    parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir)
    parser.add_argument('--valid_dir', type=str, default=test_dir)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test'])

    opt = parser.parse_args()

    torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    torch.backends.cudnn.benchmark = False

    print(opt)

    ###here for same result#####
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False

    if torch.cuda.is_available():
    	print('Setting GPU')
    	print('===> CUDA Available: ', torch.cuda.is_available())
    	opt.device = 'cuda'

    model = torch.load('')
    test_data_loader = get_dataloader(opt, 'test')

    loss_function = nn.CrossEntropyLoss()
    metric = AccumulatedAccuracyMetric()

    opt.mode = 'test'
    test_acc, test_loss = valid(opt,model, test_data_loader, metric)



