import torchvision
import torch
from torch import nn
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import *
import MyDataloader
from TumorNetwithoutSource import *

import csv
import pandas as pd
import json
import scipy.ndimage as ndimage
import SimpleITK as sitk
from medpy import metric
import numpy as np
import time
import shutil
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from visdom import Visdom
viz = Visdom(env='PiaNet TumorNet without Source 134')
viz.line([0], [0], win='train')
viz.line([0], [0], win='valid')
viz.line([0], [0], win='tumor')
#################initialization network##############
def weights_init(model):
	if isinstance(model, nn.Conv3d) or isinstance(model, nn.ConvTranspose3d):
		nn.init.kaiming_uniform_(model.weight.data, 0.25)
		nn.init.constant_(model.bias.data, 0)
	# elif isinstance(model, nn.InstanceNorm3d):
	# 	nn.init.constant_(model.weight.data,1.0)
	# 	nn.init.constant_(model.bias.data, 0)

def train_valid_seg():
	##########hyperparameters##########
	if_test = True
	if_resume = True
	max_epoches = 100
	episode = 10
	train_batch_size = 1; valid_batch_size = 1; test_batch_size = 1
	# channels=1; depth=16; height=48; width=48
	channels = 1; depth=64; height=256; width=256
	learning_rate = 0.00001
	weight_decay = 1e-4
	config = {
		# 'model':'USNETres',
		'train_csv_list': ['./TVTcsv_tumor/train' + str(i) + '.csv' for i in range(episode)],
		'valid_csv': './TVTcsv_tumor/valid_tumor.csv',
		'test_json': "/home/lihuiyu/Code/LiTS_Preprocess/test_files.json",
		#below is the saved path
		'ckpt_dir': './results/',
		'saved_dir':"/home/lihuiyu/Data/LiTS/segResults/",
		'model_dir' :"./results/20190924-083751/model_0-9.pth"
	}
	##########hyperparameters##########
	# refresh save dir
	exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
	ckpt_dir = os.path.join(config['ckpt_dir'] + exp_id)
	saved_dir = os.path.join(config['saved_dir'] + exp_id)
	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir)
	if not os.path.exists(saved_dir):
		os.makedirs(saved_dir)

	logfile = os.path.join(ckpt_dir, 'log')
	# if not if_test:
	# redirect print output to log file
	sys.stdout = Logger(logfile)#see utils.py

	###############GPU,Net,optimizer,scheduler###############
	torch.manual_seed(0)
	if torch.cuda.is_available():
		net = PiaNet().cuda()#need to do this before constructing optimizer
		loss = MTLloss().cuda()
	else:
		net = PiaNet()
		loss = MTLloss()
	cudnn.benchmark = True  # True
	# net = DataParallel(net).cuda()
	# optimizer = torch.optim.SGD(net.parameters(), learning_rate, momentum=0.9,weight_decay=weight_decay)#SGD+Momentum
	optimizer = torch.optim.Adam(net.parameters(), learning_rate, (0.9, 0.999), eps=1e-08,weight_decay=weight_decay)  # weight_decay=2e-4
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1)#decay the learning rate after 100 epoches
	###############resume or initialize prams###############
	if if_test or if_resume:
		print('if_test:',if_test,'if_resume:',if_resume)
		checkpoint = torch.load(config['model_dir'])
		net.load_state_dict(checkpoint)
	else:
		print('weight initialization')
		net.apply(weights_init)

	#test
	if test:
		val_loader = DataLoader(MyDataloader.LiTSDataloader(dir_csv=config['valid_csv'],
															channels=channels, depth=depth, height=height, width=width),
								batch_size=valid_batch_size, shuffle=False, pin_memory=True)
		valid_loss, valid_tumor, valid_iter = test(val_loader, net, loss, saved_dir)
		valid_avgloss = sum(valid_loss) / valid_iter
		valid_avgtumor = sum(valid_tumor) / valid_iter
		print("valid_loss:%.4f, valid_tumor:%.4f, Time:%.3fmin " %
			  (valid_avgloss, valid_avgtumor, (time.time() - start_time) / 60))
		# print:lr,epoch/total,loss123,accurate,time
		return

	# val_set_loader
	val_loader = DataLoader(MyDataloader.LiTSDataloader(dir_csv=config['valid_csv'],
							channels=channels, depth=depth, height=height, width=width),
							batch_size=valid_batch_size, shuffle=False,pin_memory=True)
	#################train-eval (epoch)##############################
	max_validtumor = 0.722
	for epoch in range(max_epoches):
		for epi in range(episode):
			# train_set_loader
			train_loader = DataLoader(MyDataloader.LiTSDataloader(dir_csv=config['train_csv_list'][epi],
									  channels=channels, depth=depth, height=height,width=width),
									  batch_size=train_batch_size, shuffle=True, pin_memory=True)
			print('######train epoch-epi', str(epoch),'-',str(epi), 'lr=', str(optimizer.param_groups[0]['lr']),'######')
			train_loss, train_tumor, train_iter = train(train_loader, net, loss, optimizer)
			scheduler.step(epoch)
			train_avgloss = sum(train_loss) / train_iter
			train_avgtumor = sum(train_tumor) / train_iter
			print("[%d-%d/%d], train_loss:%.4f,train_tumor:%.4f, Time:%.3fmin" %
				  (epoch, epi, max_epoches-1, train_avgloss, train_avgtumor, (time.time() - start_time) / 60))

			print('######valid epoch-epi', str(epoch),'-',str(epi),'######')
			valid_loss, valid_tumor, valid_iter = validate(val_loader, net, loss, epoch, episode, saved_dir)
			valid_avgloss = sum(valid_loss) / valid_iter
			valid_avgtumor = sum(valid_tumor) / valid_iter
			print("[%d-%d/%d], valid_loss:%.4f, valid_tumor:%.4f, Time:%.3fmin " %
				  (epoch, epi, max_epoches, valid_avgloss, valid_avgtumor, (time.time() - start_time) / 60))
			# print:lr,epoch/total,loss123,accurate,time

			#if-save-model:
			if max_validtumor < valid_avgtumor:
				max_validtumor = valid_avgtumor
				print(max_validtumor)
				state = {
					'epoche':epoch,
					'arch':str(net),
					'state_dict':net.state_dict(),
					'optimizer':optimizer.state_dict()
					#other measures
				}
				torch.save(state,ckpt_dir+'/checkpoint.pth.tar')
				#save model
				model_filename = ckpt_dir+'/model_'+str(epoch)+'-'+str(epi)+'-'+str(max_validtumor)[:6]+'.pth'
				torch.save(net.state_dict(),model_filename)
				print('Model saved in',model_filename)
			viz.line([train_avgloss], [epoch * 10 + epi], win='train', opts=dict(title='train avgloss'), update='append')
			viz.line([valid_avgloss], [epoch * 10 + epi], win='valid', opts=dict(title='valid avgloss'), update='append')
			viz.line([valid_avgtumor], [epoch * 10 + epi], win='tumor', opts=dict(title='valid avgtumor'), update='append')

def train(data_loader, net, loss, optimizer):
	net.train()#swithch to train mode
	x = []
	epoch_loss = []
	epoch_liver = []
	epoch_tumor = []
	total_iter = len(data_loader)
	for i, (data,target,origin,direction,space,ct_name) in enumerate(data_loader):
		if torch.cuda.is_available():
			data = data.cuda()
			target = target.cuda()
		output2 = net(data)
		loss_output = loss(output2, target)
		tumor_dice = Dice(output2, target)
		optimizer.zero_grad()#set the grade to zero
		loss_output.backward()
		optimizer.step()

		x.append(i)
		epoch_loss.append(loss_output.item())  # Use tensor.item() to convert a 0-dim tensor to a Python number
		epoch_tumor.append(tumor_dice)
		print("[%d/%d], loss:%.4f, tumor_dice:%.4f, name:%s" % (i, total_iter, loss_output.item(),tumor_dice,ct_name))
	# viz.line(Y=np.column_stack((epoch_loss, epoch_tumor)),
	# 		 X=np.column_stack((x, x, x)), win='loss-dice',
	# 		 opts=dict(legend=['loss', 'liver', 'tumor']),update='new')
	return epoch_loss,epoch_tumor,total_iter

def validate(data_loader, net, loss, epoch, episode, saved_dir):
	net.eval()
	epoch_loss = []
	epoch_tumor = []
	total_iter = len(data_loader)

	with torch.no_grad():#no backward
		for i, (data,target,origin,direction,space,ct_name) in enumerate(data_loader):
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
			output2 = net(data)
			loss_output = loss(output2,target)
			tumor_dice= Dice(output2, target)

			epoch_loss.append(loss_output.item())#Use tensor.item() to convert a 0-dim tensor to a Python number
			epoch_tumor.append(tumor_dice)
			print("[%d/%d], loss:%.4f, tumor_dice:%.4f, name:%s" % (i, total_iter, loss_output.item(), tumor_dice,ct_name))
			if epoch % episode == 0:
				output2_name = os.path.join(saved_dir,'valid-' + ct_name[0] + '-' + str(epoch) + '-' + str(episode) + '-output2' + '.nii')
				saved_preprocessed(output2, origin, direction, space, output2_name)
				print(output2_name)

	return epoch_loss, epoch_tumor, total_iter

def test(data_loader, net, loss, saved_dir):
	net.eval()
	epoch_loss = []
	epoch_tumor = []
	total_iter = len(data_loader)
	with torch.no_grad():#no backward
		for i, (data,target,origin,direction,space,ct_name) in enumerate(data_loader):
			if torch.cuda.is_available():
				data = data.cuda()
				target = target.cuda()
			output2 = net(data)
			loss_output = loss(output2,target)
			tumor_dice= Dice(output2, target)

			epoch_loss.append(loss_output.item())#Use tensor.item() to convert a 0-dim tensor to a Python number
			epoch_tumor.append(tumor_dice)
			print("[%d/%d], loss:%.4f, tumor_dice:%.4f, name:%s" % (i, total_iter, loss_output.item(), tumor_dice,ct_name))

			output2_name = os.path.join(saved_dir,'valid-' + ct_name[0] + '-output2.nii')
			saved_preprocessed(output2, origin, direction, space, output2_name)
			print(output2_name)

	return epoch_loss, epoch_tumor, total_iter

if __name__ == '__main__':
	# print(torch.__version__)#0.4.1
	print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))
	start_time = time.time()
	train_valid_seg()
	print('Time {:.3f} min'.format((time.time() - start_time) / 60))
	print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))