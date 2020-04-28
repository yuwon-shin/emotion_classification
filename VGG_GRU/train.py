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
import numpy
import dataset
import random
from utils import *
from VGG_gru import FERANet
# from tensorboardX import SummaryWriter
import argparse
import subprocess
import math

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

parser = argparse.ArgumentParser(description='PyTorch Facial Expression')

parser.add_argument('--batch_size', type=int, default=1, metavar='N',
					help='input batch size for training (max: 3)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
					help='number of epochs to train (default: 150)')
# parser.add_argument('--length', type=int, default=16,
# 					help='crop length for training')

parser.add_argument('--preInitial', type=bool, default=True,
					help='initial vgg from ImageNet')

parser.add_argument('--dataset', type=str, default="face_data",
					help='train dataset')
parser.add_argument('--checkpoint_dir', type=str, default="checkpoint",
					help='checkpoint direction')
args = parser.parse_args()

print(args)
print(args.dataset)
trainlist = "C:/Users/신유원/Desktop/capstone project/workspaces/emotion_classification/VGG_GRU/train/"
vallist =  "C:/Users/신유원/Desktop/capstone project/workspaces/emotion_classification/VGG_GRU/test/"
# if os.path.isfile != True:
# 	subprocess.call(["python", "./TrainTestlist/"+args.dataset+"/getTraintest_"+args.dataset+".py"])


backupdir = "weight"
batch_size = 1
learning_rate = 0.00001

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


processed_batches = 0
kwargs = {'num_workers': num_workers, 'pin_memory': True}
# model = model.cuda()        #non gpu


optimizer = optim.SGD(model.parameters(), lr=learning_rate , momentum=0.9, weight_decay= 0.00005)

loss_function = nn.CrossEntropyLoss()

use_Tensorboard = False


def save_checkpoint(epoch, loss):
	checkpoint_dir = os.path.abspath(args.checkpoint_dir)
	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	checkpoint_path = os.path.join(checkpoint_dir, "models_epoch_%04d_loss_%.6f.pth" % (epoch, loss))
	torch.save(checkpoint_path)
	print("Checkpoint saved to {}".format(checkpoint_path))


def train(epoch,optimizer):
	if not os.path.exists(args.checkpoint_dir):
		os.makedirs(args.checkpoint_dir)

	train_loader = torch.utils.data.DataLoader(
		dataset.listDataset(trainlist,                 #length = args.length,
					  #shuffle=True,
					   train=True,
					   dataset = args.dataset),
		batch_size=args.batch_size, shuffle=False,  **kwargs)

	for param_group in optimizer.param_groups:
		train_learning_rate = float(param_group['lr'])

	logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), train_learning_rate))

	running_loss = 0.0

	model.train()

	for batch_idx, (data, label) in enumerate(train_loader):


		data = data.squeeze(0)
		data = Variable(data)#.cuda()

		label = Variable(label.long())#.cuda()
		label = label.squeeze(1)

		optimizer.zero_grad()

		output = model(data)

		loss = loss_function(output, label)

		running_loss += loss.data[0]

		loss.backward()

		optimizer.step()

	if epoch %1 == 0:
		logging('Loss:{:.6f}'.format(running_loss))


def eval(epoch,metric):

	model.eval()

	global best_accuracy

	metric.reset()

	test_loader = torch.utils.data.DataLoader(
		dataset.listDataset(vallist,length = args.length,
					shuffle=False,
					train=False,
					dataset = args.dataset),
					batch_size=1, shuffle=False, **kwargs)


	for batch_idx, (data, label) in enumerate(test_loader):

		data = data.squeeze(0)

		Batch,T,C,H,W = data.size()

		data = Variable(data,volatile=True).cuda()

		label = Variable(label.long(),volatile=True).cuda()
		label = label.squeeze(1)

		output = []
		for batch_index in range(Batch):
			output_feature = model(data[batch_index])
			output.append(output_feature)

		output = torch.mean(torch.cat(output), 0, keepdim=True)

		metric(output, label)
		

	if accuracy >= best_accuracy:
		best_accuracy = accuracy
		print("saving accuracy is: ",accuracy)
		torch.save(model.state_dict(),'%s/model_%d.pkl' % (backupdir,epoch))

	logging("test accuracy: %f" % (accuracy))
	logging("best accuracy: %f" % (best_accuracy))

	return accuracy

for epoch in range(1, args.epochs+1): 

	train(epoch,optimizer)
	# eval_accuary = eval(epoch,metric)
	# save_checkpoint(epoch, eval_accuracy)
