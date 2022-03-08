#ArchillesEncoder
#example command:
#python3 /home2/twang6/projects/benisse/Benisse/AchillesEncoder.py \
#--input_data /home2/twang6/projects/benisse/Benisse/example/10x_NSCLC.csv \
#--output_data /home2/twang6/projects/benisse/Benisse/example/encoded_10x_NSCLC.csv \
#--encode_dim 40

import argparse
import csv
import os
# Load modules
# !usr/bin/env python3
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/CMC")
from model_util import MyAlexNetCMC
from contrast_util import NCEAverage, NCESoftmaxLoss
from data_pre import load_BCRdata2, datasetMap_nt
from data_util import Dataset

def strtobool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Please input True or False')

#Get parameters
def parse_option():
    parser=argparse.ArgumentParser('Arguments for training')
    parser.add_argument('--input_data',type=str,help='BCR sequence data file (.csv) to be embedded, \
    	contains at least two columns in the names of "contigs" (unique identifiers of cells) and "cdr3" \
    	(BCR b-chain CDR3 sequences), or \
    	a folder path where contains all and only BCR sequence data files. Output will be concatenate into \
    	one output file.')
    parser.add_argument('--atchley_factors',type=str,help='File that saved the Atchely factors')
    parser.add_argument('--model',default=None,type=str,help='The trained model to apply (default: none)')
    parser.add_argument('--output_data',type=str,help='Encoded BCR sequences, the output data file')
    parser.add_argument('--encode_dim',type=int,default=40,help='Columns of padded atchley matrix (default: 80)')
    parser.add_argument('--pad_length',type=int,default=130,help='Length of padded nucleotide sequence (default: 130), \
    	cells with nucleotide sequences longer than the length will be removed.')
    parser.add_argument('--cuda', default=True, type=strtobool, help='Whether to use CUDA to train model')
    opt=parser.parse_args()
    return opt

#Load data
opt=parse_option()
test=load_BCRdata2(opt)

#Data prep
aa_dict=dict()
with open(os.path.dirname(os.path.abspath(__file__))+"/dependency/Atchley_factors.csv",'r') as aa:
    aa_reader=csv.reader(aa)
    next(aa_reader, None)
    for rows in aa_reader:
        aa_name=rows[0]
        aa_factor=rows[1:len(rows)]
        aa_dict[aa_name]=np.asarray(aa_factor,dtype='float')
cdr_test,vdj_test,cdr3_seq_test=datasetMap_nt(test,aa_dict,opt.encode_dim,opt.pad_length)
batch_size = 64
random_seed= 123
test_indices = list(set(vdj_test.keys()))
cdr_shape = cdr_test[list(cdr_test.keys())[0]].shape[0]
n_vdj = vdj_test[list(vdj_test.keys())[0]].size()[0]
n_data = len(test_indices)
n_out_features = 20

test_set = Dataset(test_indices,cdr_test,vdj_test,cdr3_seq_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, 
                                          shuffle=False, sampler=None,batch_sampler=None,num_workers=1)

#Load model
if torch.cuda.is_available():
    if opt.cuda:
        device = torch.device("cuda")
    if not opt.cuda:
        print("Note: You have CUDA enabled but not using it.")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

feat_dim=20
in_feature=130
nce_k = 1
nce_t = 0.2
nce_m = 0.9
lr = 0.001
momentum = 0.9
weight_decay = 0.0001
gradient_clip = 5

if opt.cuda:
    state=torch.load(os.path.dirname(os.path.abspath(__file__))+"/dependency/trained_model.pt")
else:
    state = torch.load(os.path.dirname(os.path.abspath(__file__)) + "/dependency/trained_model.pt", map_location=torch.device('cpu'))
test_model=MyAlexNetCMC(in_feature=in_feature,feat_dim=feat_dim,freeze=True).to(device)
contrast = NCEAverage(n_out_features, n_data, nce_k, device, nce_t, nce_m).to(device)
criterion_cdr = NCESoftmaxLoss().to(device)
criterion_vdj = NCESoftmaxLoss().to(device)
optimizer = torch.optim.SGD(test_model.parameters(),
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
test_model.load_state_dict(state['model'])
optimizer.load_state_dict(state['optimizer'])
test_model.eval()

#Calculate 
def embed(test_loader, model):

    for idx, (data, index) in enumerate(test_loader):
        index = index.to(device)
        for _ in list(data.keys())[0:2]:
            data[_] = data[_].float().to(device)
        feat_cdr,feat_vdj,cdr3_seq = model(data)
        if idx==0:
            feature_array=pd.DataFrame(feat_cdr.cpu().numpy())
            feature_array['index']=cdr3_seq
        else:
            feature_array_tmp=pd.DataFrame(feat_cdr.cpu().numpy())
            feature_array_tmp['index']=cdr3_seq
            feature_array=feature_array.append(feature_array_tmp)

    feature_array.sort_values(by=['index'], inplace = True)

    return feature_array

encoded_BCR=embed(test_loader,test_model)
encoded_BCR.to_csv(opt.output_data,sep=',')

# Ze, we keep you in our heart forever
