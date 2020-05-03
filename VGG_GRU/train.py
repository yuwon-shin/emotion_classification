from __future__ import print_function

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import models
import numpy as np
import dataset
import random
from utils import *
from data_loader import FACEDATA, get_train_valid_dataloader
from VGG_gru import FERANet
# from tensorboardX import SummaryWriter
# import model as md
import argparse
import subprocess
import math

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
data_dir = '../../../data/face_data'
parser = argparse.ArgumentParser(description='PyTorch Facial Expression')

parser.add_argument('--batch_size', type=int, default=16,
					help='input batch size for training (max: 3)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
					help='number of epochs to train (default: 150)')
parser.add_argument('--img_size', type=int, default=64,
					help='input image size')

parser.add_argument('--preInitial', type=bool, default=True,
					help='initial vgg from ImageNet')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test'])
parser.add_argument('--dataset_dir', type=str, default=data_dir,
					help='train dataset')
parser.add_argument('--checkpoint_dir', type=str, default="checkpoint",
					help='checkpoint direction')
parser.add_argument('--train_ratio', type=float, default=0.95)
args = parser.parse_args()

print(args)
# print(args.dataset_dir)
train_dataloader, valid_dataloader = get_train_valid_dataloader(args)
# trainlist = os.path.abspath(args.dataset+"/train")
# vallist =  os.path.abspath(args.dataset+"/val")
# if os.path.isfile != True:
# 	subprocess.call(["python", "./TrainTestlist/"+args.dataset+"/getTraintest_"+args.dataset+".py"])


backupdir = "weight"
batch_size = 1
learning_rate = 0.00001
epoch = args.epochs
best_accuracy = 0.
metric = AccumulatedAccuracyMetric()

####here for same result#####
num_workers = 0
torch.backends.cudnn.enabled = False
torch.manual_seed(1)
# torch.cuda.manual_seed(1)
random.seed(1)
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
if args.preInitial == False:
	model = FERANet()
	model = Initial(model)
else:
	model = FERANet()

num_classes = 3
# model = md.init_pretrained_models('vgg16', num_classes, feature_extract=True, use_pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(params = model.parameters(),lr=learning_rate,betas=(0.9,0.999), eps=1e-8, weight_decay=0.00005)
loss_function = nn.CrossEntropyLoss()

use_Tensorboard = False


def save_checkpoint(epoch, loss):
	checkpoint_dir = os.path.abspath(args.checkpoint_dir)
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	checkpoint_path = os.path.join(checkpoint_dir, "models_epoch_%04d_loss_%.6f.pth" % (epoch, loss))
	torch.save(checkpoint_path)
	print("Checkpoint saved to {}".format(checkpoint_path))


def train(epoch,optimizer,train_loader):
	if not os.path.exists(args.checkpoint_dir):
		os.makedirs(args.checkpoint_dir)

	# train_loader = torch.utils.data.DataLoader(
	# 	FACEDATA(trainlist,args, train = True),
	# 	batch_size=args.batch_size, shuffle=True,  **kwargs)

	for param_group in optimizer.param_groups:
		train_learning_rate = float(param_group['lr'])

	logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), train_learning_rate))

	running_loss = 0.0

	model.train()

	for batch_idx, (data, label) in enumerate(train_loader):


		# data = data.squeeze(0)
		data = Variable(data)#.cuda()
		print(data.shape)
		label = Variable(label.long())#.cuda()
		labels = label.argmax(1)
		# label = label.squeeze(1)
		print(labels.shape)
		optimizer.zero_grad()  #set the gradients to zero before starting to do backpropragation

		output = model(data)
		output = output.squeeze(0)
		print(output.shape)
		loss = loss_function(output, label)

		running_loss += loss.data

		loss.backward()

		optimizer.step()

	if epoch %1 == 0:
		logging('Loss:{:.6f}'.format(running_loss))


def eval(epoch,metric, valid_loader):

	model.eval()

	global best_accuracy

	metric.reset()

	# test_loader = torch.utils.data.DataLoader(
	# 	dataset.listDataset(vallist,length = args.length,
	# 				shuffle=False,
	# 				train=False,
	# 				dataset = args.dataset),
	# 				batch_size=1, shuffle=True, **kwargs)

	for batch_idx, (data, label) in enumerate(valid_loader):

		data = data.squeeze(0)

		Batch,T,C,H,W = data.size()

		data = Variable(data, volatile=True)#.cuda()

		label = Variable(label.long(), volatile=True)#.cuda()
		label = label.squeeze(1)

		output = []
		for batch_index in range(Batch):
			output_feature = model(data[batch_index])
			output.append(output_feature)

		output = torch.mean(torch.cat(output), 0, keepdim=True)

		metric(output, label)
		accuracy, eval_loss = metric.value()

	if accuracy >= best_accuracy:
		best_accuracy = accuracy
		print("saving accuracy is: ",accuracy)
		torch.save(model.state_dict(),'%s/model_%d.pkl' % (backupdir,epoch))

	logging("test accuracy: %f" % (accuracy))
	logging("best accuracy: %f" % (best_accuracy))
	logging("train eval loss: %f" % (eval_loss))

	return accuracy, eval_loss


for epoch in range(1, args.epochs+1):

	train(epoch,optimizer,train_dataloader)
	if (epcch+1)%5 == 0:
		eval_accuary, eval_loss = eval(epoch,metric,valid_dataloader)
	save_checkpoint(epoch, eval_accuracy)
