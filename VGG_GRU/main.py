## cuda 사용

from __future__ import print_function	# for compatibility
import os
import time
import numpy
import random
import dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import models
from utils import *    # *: utils 안의 모든 이름에 access
from VGG_gru import VggNet
# from tensorboardX import SummaryWriter
import argparse
import subprocess    # 다른 언어로 만들어진 프로그램을 통합 ex) for 오래된 os.*
from tqdm import tqdm
from data_loader import FER, get_dataloader

def train(opt, epoch, model, optimizer, loss_function, train_loader):

	start_time = time.time()

	for param_group in optimizer.param_groups:
		train_learning_rate = float(param_group['lr'])

	# logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), train_learning_rate))

	running_loss = 0.0

	model.train()
	print('[*]Training...')
	print('Epoch {}/{}'.format(epoch, opt.epochs))
	for batch_idx, (data, label) in enumerate(tqdm(train_loader)):


		data = data.squeeze(0)
		data = Variable(data).to(opt.device)

		label = Variable(label.long()).to(opt.device)
		label = label.squeeze(1)		
		# print('main.py label: ',label)
		# print("main.py size(label): ",label.shape)
		
		optimizer.zero_grad()

		output = model(data)
		# print("main.py output : ",output)
		# print('main.py output.shape  :', output.shape)

		loss = loss_function(output, label)
		# print('loss.item()',loss.item())

		running_loss += loss.item()
		print(' Training [{}/{}] {:.2f}sec.. loss : {:.6f}, running_loss : {:.6f}'.format(batch_idx,len(train_loader), time.time()-start_time, loss.item(), running_loss/(batch_idx+1)))

		loss.backward()

		optimizer.step()
	
	print('==> total {:.2f}sec.. Loss:{:.6f}'
				.format(epoch, opt.epochs, time.time()-start_time, running_loss/(batch_idx+1)))


def valid(opt, epoch, model, valid_loader, metric):

	start_time = time.time()

	metric.reset()

	model.eval()

	print('[*]Validation...')
	print('Epoch {}/{}'.format(epoch, opt.epochs))

	with torch.no_grad():
		for batch_idx, (data, label) in enumerate(tqdm(valid_loader)):

			Batch,T,C,H,W = data.size()
			
			data = data.squeeze(0) 
			data = Variable(data).to(opt.device)


			label = Variable(label.long()).to(opt.device)
			label = label.squeeze(1)

			output = []
			for batch_index in range(Batch):
				# print('batch: ',Batch)
				# print('batch idx: ',batch_index)
				# print('@@ ',data[batch_index].shape)
				output_feature = model(data[batch_index])
				output.append(output_feature)
			# print('output: ',output)

			output = torch.cat(output,0)
			# print('concat output :',output)
			
			metric(output, label)
			accuracy,eval_loss = metric.value()
			avg_loss = eval_loss/((batch_idx+1)*opt.batch_size)
			
			print('Validation [{}/{}] accuracy : {:.2f}, loss : {:.6f}'.format(batch_idx, len(valid_loader), accuracy, avg_loss))
			print('acc & loss:',accuracy, '&', eval_loss)

	print('[*]Validation...')
	print('Epoch {}/{}'.format(epoch, opt.epochs))

	return accuracy, avg_loss	


if __name__ == "__main__":
	# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

	data_dir = '../../../../data/face_data'
	train_dir = os.path.join(data_dir, 'train')
	checkpoint_dir = os.path.join(data_dir, 'checkpoint')
	test_dir = os.path.join(data_dir, 'val')
	result_dir = os.path.join(data_dir, 'result')

	parser = argparse.ArgumentParser(description='PyTorch Facial Expression')

	parser.add_argument('--batch_size', type=int, default=16,
						help='input batch size for valid/train = 1')
	parser.add_argument('--img_size', type=int, default=64)
	parser.add_argument('--epochs', type=int, default=200,
						help='number of epochs to train (default: 200)')
	parser.add_argument('--start_epoch', default=1, type=int)
	parser.add_argument('--iter', type=int, default=0,
						help='number of iters for each epoch, if iter is 0, it means number of images')					
	parser.add_argument('--length', type=int, default=4,
						help='data shape : (b, <<l>>, c, h, w) | meaning batch in training | for making each batch containing same class')
	parser.add_argument('--lr', type=float, default=1e-05)
	parser.add_argument('--num_workers', type=int, default=2)
	parser.add_argument('--optim', type=str, default='sgd', choices=['adam', 'sgd'])

	parser.add_argument('--multi_gpu', default=False, action='store_true',
						help='Use Multi GPU')

	#####
	# parser.add_argument('--no_multi_gpu', default=False, action='store_true',
	#					help='Do Not Use Multi GPUs')
	# parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')

	parser.add_argument('--data_dir', type=str, default=data_dir,
						help='dataset path')
	parser.add_argument('--train_dir', type=str, default=train_dir)
	parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir)
	parser.add_argument('--valid_dir', type=str, default=test_dir)
	parser.add_argument('--result_dir', type=str, default=result_dir)

	parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test'])
	parser.add_argument('--resume', action='store_true', default=False)
	parser.add_argument('--resume_best', action='store_true', default=False)

	opt = parser.parse_args()

	# torch.manual_seed(1)
	torch.cuda.manual_seed(1)
	torch.backends.cudnn.benchmark = False

	print(opt)

	###here for same result#####
	torch.backends.cudnn.enabled = False
	torch.backends.cudnn.deterministic = False

	if torch.cuda.is_available():
		print('Setting GPU')
		print('===> CUDA Available: ', torch.cuda.is_available())
		opt.device = 'cuda'

		if torch.cuda.device_count() > 1 and not opt.no_multi_gpu:
			print('===> Use {} Multi GPUs'.format(torch.cuda.device_count()))
		else :
			opt.multi_gpu = False

	else : 
		print('Using only CPU')

	print('Initialize networks')
	model = VggNet()
	model = model.to(opt.device)

	print("Setting Optimizer & loss")
	if opt.optim == 'sgd':
		optimizer = optim.SGD(model.parameters(), lr=opt.lr , momentum=0.9, weight_decay= 0.00005)
	else :
		optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9,0.999), eps=1e-8, weight_decay=0.00005)
	
	loss_function = nn.CrossEntropyLoss()

	if opt.resume or opt.resume_best:
		opt.start_epoch, model, optimizer = load_model(opt, model, optimizer=optimizer)
	
	if opt.multi_gpu:
		model = nn.DataParallel(model)

	train_data_loader = get_dataloader(opt,'train')
	valid_data_loader = get_dataloader(opt,'valid')

	metric = AccumulatedAccuracyMetric()
	pre_valid_loss = float('inf')

	for epoch in range(opt.start_epoch, opt.epochs+1): 
		opt.mode = 'train'
		train(opt, epoch, model, optimizer, loss_function, train_data_loader)

		opt.mode = 'valid'
		valid_acc, valid_loss = valid(opt, epoch, model, valid_data_loader, metric)

		if pre_valid_loss > valid_loss:
			pre_valid_loss = valid_loss
			save_checkpoint(opt, model, optimizer, epoch, valid_loss, valid_acc)
		# eval_accuary,eval_loss = eval(epoch,metric)
