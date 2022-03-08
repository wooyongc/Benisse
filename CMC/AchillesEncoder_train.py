#!usr/bin/env python3 in conda env torch_scBCR
#Adapted from CMC: https://github.com/HobbitLong/CMC
#Example code:
#python3 AchillesEncoder.py --input_data cleaned_BCRmltrain \
#--atchley_factors Atchley_factors.csv \
#--resume model_BCRmltrain \
#--break_point model_BCRmltrain/epoch_25.pt \
#--encode_dim 40 --pad_length 130

from __future__ import print_function
import sys
import os
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import socket
import pandas as pd
import csv
import numpy as np
import pickle
import re
from model_util import MyAlexNetCMC
from contrast_util import NCEAverage,AverageMeter,NCESoftmaxLoss
from torch.utils.data.sampler import SubsetRandomSampler
from data_pre import load_BCRdata,aamapping,datasetMap_nt,ntmapping
from data_util import Dataset
from random import seed,sample
from sklearn.metrics import roc_curve,auc

def strtobool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_option():
    parser=argparse.ArgumentParser('Arguments for training')
    parser.add_argument('--input_data',type=str,help='Folder that saved data used for training')
    parser.add_argument('--atchley_factors',type=str,help='File that saved the Atchely factors')
    parser.add_argument('--resume',default='',metavar='PATH',help='Path to save the latest checkpoint (default: none)')
    parser.add_argument('--break_point',default=None,type=str,help='The latest checkpoint file to load (default: none)')
    parser.add_argument('--encode_dim',type=int,default=40,help='Columns of padded atchley matrix (default: 80)')
    parser.add_argument('--pad_length',type=int,default=130,help='Length of padded nucleotide sequence (default: 130)')
    parser.add_argument('--cuda', default=True, type=strtobool, help='Whether to use CUDA to train model')
    opt=parser.parse_args()
    return opt

# Load data
opt=parse_option()
full=load_BCRdata(opt)
aa_dict=dict()
with open(opt.atchley_factors,'r') as aa:
    aa_reader=csv.reader(aa)
    next(aa_reader, None)
    for rows in aa_reader:
        aa_name=rows[0]
        aa_factor=rows[1:len(rows)]
        aa_dict[aa_name]=np.asarray(aa_factor,dtype='float')
cdr_full,vdj_full,cdr3_seq_full=datasetMap_nt(full,aa_dict,opt.encode_dim,opt.pad_length)

# Generator
batch_size = 512
indices = list(set(vdj_full.keys()))
test_split = 0.01
random_seed= 123
split = int(np.floor(test_split * len(indices)))
shuffle_dataset=True
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

#NCE average parameter
if torch.cuda.is_available():
    if args.cuda:
        device = torch.device("cuda")
    if not args.cuda:
        print("Note: You have CUDA enabled but not using it.")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

cdr_shape = cdr_full[list(cdr_full.keys())[0]].shape[0]
n_data = len(train_indices)
n_data_test = len(test_indices)
n_out_features = 20

train_set = Dataset(train_indices,cdr_full,vdj_full,cdr3_seq_full)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                                          shuffle=True, num_workers=1)

test_set = Dataset(test_indices,cdr_full,vdj_full,cdr3_seq_full)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, 
                                          shuffle=True, num_workers=1)

# NCE parameters
nce_k = 1
nce_t = 0.2
nce_m = 0.9
feat_dim = n_out_features
in_feature=vdj_full[list(vdj_full.keys())[0]].size()[0]

# Training parameters
lr = 0.001
momentum = 0.9
weight_decay = 0.0001
gradient_clip = 5

# Set model
model = MyAlexNetCMC(in_feature=in_feature,feat_dim=feat_dim).to(device)
contrast = NCEAverage(n_out_features, n_data, nce_k, device, nce_t, nce_m).to(device)
criterion_cdr = NCESoftmaxLoss().to(device)
criterion_vdj = NCESoftmaxLoss().to(device)

# Set optimizer
optimizer = torch.optim.SGD(model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)

#Optional: resume training from the latest break point
model_file = opt.break_point
if model_file is not None:
    print("=> loading checkpoint '{}'".format(model_file))
    checkpoint = torch.load(model_file, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    contrast.load_state_dict(checkpoint['contrast'])
    del checkpoint
else:
    start_epoch = 0

#Training function
def train(epoch, train_loader, model, contrast, criterion_cdr, criterion_vdj, optimizer, 
          gradient_clip=10, print_freq=1):
    """
    One epoch training
    """
    model.train()
    contrast.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cdr_loss_meter = AverageMeter()
    vdj_loss_meter = AverageMeter()
    cdr_prob_meter = AverageMeter()
    vdj_prob_meter = AverageMeter()

    end = time.time()
    for idx, (data, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        batch_size = data['cdr'].size(0)
        index = index.to(device)
        for _ in data.keys():
            data[_] = data[_].float().to(device)
        # ===================forward=====================
        feat_cdr, feat_vdj,cdr3_seq = model(data)
        out_cdr, out_vdj = contrast(feat_cdr, feat_vdj, index)
        
        cdr_loss = criterion_cdr(out_cdr)
        vdj_loss = criterion_vdj(out_vdj)
        cdr_prob = out_cdr[:, 0].mean()
        vdj_prob = out_vdj[:, 0].mean()

        loss = cdr_loss+vdj_loss
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        torch.nn.utils.clip_grad_norm_(contrast.parameters(), gradient_clip)
        optimizer.step()
        # ===================meters=====================
        losses.update(loss.item(), batch_size)
        cdr_loss_meter.update(cdr_loss.item(), batch_size)
        cdr_prob_meter.update(cdr_prob.item(), batch_size)
        vdj_loss_meter.update(vdj_loss.item(), batch_size)
        vdj_prob_meter.update(vdj_prob.item(), batch_size)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'cdr_p {cdr_probs.val:.3f} ({cdr_probs.avg:.3f})\t'
                  'vdj_p {vdj_probs.val:.3f} ({vdj_probs.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, cdr_probs=cdr_prob_meter,
                   vdj_probs=vdj_prob_meter))
            sys.stdout.flush()

    return cdr_loss_meter.avg, cdr_prob_meter.avg, vdj_loss_meter.avg, vdj_prob_meter.avg

#Acc and roc_score
def predict(test_loader, model, contrast,criterion_cdr,criterion_vdj):
    acc=dict()
    roc_score=dict()
    model.eval()
    contrast.eval()
    with torch.no_grad():
        for idx, (data, index) in enumerate(test_loader):
            index = index.to(device)
            for _ in list(data.keys())[0:2]:
                data[_] = data[_].float().to(device)
            feat_cdr,feat_vdj,cdr3_seq = model(data)
            out_cdr, out_vdj = contrast(feat_cdr, feat_vdj, index)
            loss_cdr=criterion_cdr(out_cdr)
            loss_vdj=criterion_vdj(out_vdj)
            loss=loss_cdr+loss_vdj
            print('Batch {0}: test loss {1:.3f}'.format(idx,loss))
            out_cdr=out_cdr.squeeze()
            out_vdj=out_vdj.squeeze()
            acc_cdr=torch.argmax(out_cdr,dim=1)
            acc_vdj=torch.argmax(out_vdj,dim=1)
            acc_vdj=acc_vdj.squeeze()
            if idx==0:
                acc['cdr']=acc_cdr
                acc['vdj']=acc_vdj
                roc_score['cdr']=out_cdr.flatten()
                roc_score['vdj']=out_vdj.flatten()
            else:
                acc['cdr']=torch.cat((acc['cdr'],acc_cdr),0)
                acc['vdj']=torch.cat((acc['vdj'],acc_vdj),0)
                roc_score['cdr']=torch.cat((roc_score['cdr'],out_cdr.flatten()),0)
                roc_score['vdj']=torch.cat((roc_score['vdj'],out_vdj.flatten()),0)
    return acc,roc_score,loss

hist = dict()
hist['cdr_loss'] = []
hist['cdr_prob'] = []
hist['vdj_loss'] = []
hist['vdj_prob'] = []
hist['test_loss'] = []

save_freq = 5

for epoch in range(start_epoch,300):
    cdr_loss, cdr_prob, vdj_loss, vdj_prob = train(epoch, train_loader, model, contrast, criterion_cdr, criterion_vdj, optimizer, 
              gradient_clip=gradient_clip, print_freq=1)
    acc,roc_score,test_loss=predict(test_loader,model,contrast,criterion_cdr,criterion_vdj)
    
    hist['cdr_loss'].append(cdr_loss)
    hist['cdr_prob'].append(cdr_prob)
    hist['vdj_loss'].append(vdj_loss)
    hist['vdj_prob'].append(vdj_prob)
    hist['test_loss'].append(test_loss)
    acc['cdr']=acc['cdr'].cpu().numpy()
    acc['vdj']=acc['vdj'].cpu().numpy()
    roc_score['cdr']=roc_score['cdr'].cpu().numpy()
    roc_score['vdj']=roc_score['vdj'].cpu().numpy()
    predict_label=np.zeros(len(roc_score['cdr']))
    predict_label[::2]=1
    fpr,tpr,_ = roc_curve(predict_label,roc_score['cdr'])
    roc_auc=auc(fpr,tpr)
    print('cdr accuracy: ')
    print(len(np.where(acc['cdr']==0)[0])/len(acc['cdr']))
    print('nt accuracy: ')
    print(len(np.where(acc['vdj']==0)[0])/len(acc['vdj']))
    print('cdr AUC: ')
    print(roc_auc)
     #Save model
    if epoch % save_freq == 0 and epoch != 0:
        print("Saving model...")
        state = { 'model': model.state_dict(),
                  'contrast': contrast.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epoch}
        torch.save(state, opt.resume+"/epoch_{}.pt".format(epoch))

torch.save(hist, opt.resume+"/hist.pt")

#post check functions
import matplotlib.pyplot as plt
def plot_loss(hist):
    plt.title('CDR loss')
    plt.plot(hist['cdr_loss'])
    plt.show()
    plt.title('NT loss')
    plt.plot(hist['vdj_loss'])
    plt.show()
    plt.title('Test loss')
    plt.plot(hist['test_loss'])
    plt.show()

#hist=torch.load(opt.resume+"/hist.pt")
#plot_loss(hist)

