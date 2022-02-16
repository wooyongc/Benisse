from __future__ import print_function
import pickle
import numpy as np
import pandas as pd
import torch


def load_BCRdata(opt):
    """From opt.input_data load and process cdr3 and vdj data
    """
    import os
    filedir=opt.input_data
    if filedir.find('.csv')>(-1):
        datasets=[filedir]
    else:
        if not os.path.exists(filedir):
            return 'ERROR: invalid file path: ' + filedir
        datasets=[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(filedir)) for f in fn]
    for index,file in enumerate(datasets):
        if index % 10==0:
            print('Reading file:')
            print(file)
        f=pd.read_csv(file,header=0)
        if index==0:
            full_data=f[['contigs','cdr3','cdr3_nt']]
        else:
            full_data=full_data.append(f[['contigs','cdr3','cdr3_nt']])
    full_data=full_data.drop_duplicates(full_data.columns.difference(['contigs']),
                     keep='first')
    full_data=full_data.dropna()
    full_data.index=range(0,len(full_data['cdr3']))
    return full_data

def load_BCRdata2(opt):
    """From opt.input_data load and process cdr3,
       generate mock nt data, only used for the prediction mode.
    """
    import os
    filedir=opt.input_data
    if filedir.find('.csv')>(-1):
        datasets=[filedir]
    else:
        if not os.path.exists(filedir):
            return 'ERROR: invalid file path: ' + filedir
        datasets=[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(filedir)) for f in fn]
    for index,file in enumerate(datasets):
        if index % 10==0:
            print('Reading file:')
            print(file)
        f=pd.read_csv(file,header=0)
        if index==0:
            full_data=f[['contigs','cdr3']]
        else:
            full_data=full_data.append(f[['contigs','cdr3']])
    #Mock cdr3_nt, for the convenience of coding, do not affect the results.
    full_data['cdr3_nt']='TGTGCGAAATCGTATAGCAGAGACCTGCCGCGGTACTTTGGCTCCTGG'
    full_data=full_data.drop_duplicates(full_data.columns.difference(['contigs']),
                     keep='first')
    full_data=full_data.dropna()
    full_data.index=range(0,len(full_data['cdr3']))
    return full_data
    
def aamapping(peptideSeq,aa_dict,encode_dim):
    #Transform aa seqs to Atchley's factors.
    peptideArray = []
    if len(peptideSeq)>encode_dim:
        print('ERROR: aa seq length: '+str(len(peptideSeq))+' is too large!')
        peptideSeq=peptideSeq[0:encode_dim]
    for aa_single in peptideSeq:
        try:
            peptideArray.append(aa_dict[aa_single])
        except KeyError:
            print('ERROR: improper aa seqs: '+peptideSeq)
            peptideArray.append(np.zeros(5,dtype='float64'))
    for i in range(0,encode_dim-len(peptideSeq)):
        peptideArray.append(np.zeros(5,dtype='float64'))
    return np.asarray(peptideArray)

def VDJmapping(VDJset,genes):
    label=torch.zeros(len(VDJset),1)
    ind=[np.where(VDJset==genes[i])[0][0] for i in range(0, len(genes))]
    for ind_tmp in ind:
        label[ind_tmp]=1
    return label

def ntmapping(nt,pad_length=130):
    '''
    Ordinal encoding for nucleotide sequences,
    a=0.25, c=0.50, g=0.75, t=1.00, n=0.00,
    padded to the max observed length 130 with 0s.
    '''
    nt=np.array(list(nt.lower()))
    label=torch.zeros(pad_length,1)
    label[np.where(nt=='a')]=0.25
    label[np.where(nt=='c')]=0.5
    label[np.where(nt=='g')]=0.75
    label[np.where(nt=='t')]=1
    return label

def datasetMap_vdj(dataset,aa_dict,VDJset,encode_dim):
    #Wrapper of aamapping
    BCR_dict=dict()
    VDJ_dict=dict()
    for i in range(0,len(dataset['cdr3'])):
        if i % 10000 == 0:
            print(i)
        if i % 1000000 == 0:
            print('save tmp files:')
            print(i)
            cdr_tmp = open('/home2/s421955/projects/scBCR/data/model_BCRmltrain/cdr_tmp.pkl',"wb")
            pickle.dump(BCR_dict,cdr_tmp)
            cdr_tmp.close()
            vdj_tmp = open('/home2/s421955/projects/scBCR/data/model_BCRmltrain/vdj_tmp.pkl',"wb")
            pickle.dump(VDJ_dict,vdj_tmp)
            vdj_tmp.close()
        BCR_key=dataset['contigs'][i]
        if BCR_key in BCR_dict.keys():
            BCR_key=BCR_key+'__'+str(i)
        BCR_dictarray=aamapping(dataset['cdr3'][i],aa_dict,encode_dim)
        BCR_dict[BCR_key]=BCR_dictarray
        VDJ_dictvector=VDJmapping(VDJset,dataset.loc[i,['v_gene','d_gene','j_gene']])
        VDJ_dict[BCR_key]=VDJ_dictvector
    return BCR_dict,VDJ_dict

def datasetMap_nt(dataset,aa_dict,encode_dim,pad_length=130):
    #Wrapper of aamapping
    BCR_dict=dict()
    nt_dict=dict()
    cdr3_seq_dict=dict()
    for i in range(0,len(dataset['cdr3'])):
        if i % 10000 == 0:
            print('Converting row:')
            print(i)
        BCR_key=dataset['contigs'][i]
        if BCR_key in BCR_dict.keys():
            BCR_key=BCR_key+'__'+str(i)
        BCR_dictarray=aamapping(dataset['cdr3'][i],aa_dict,encode_dim)
        BCR_dict[BCR_key]=BCR_dictarray
        nt_dictvector=ntmapping(dataset.loc[i,'cdr3_nt'],pad_length)
        nt_dict[BCR_key]=nt_dictvector
        cdr3_seq_dict[BCR_key]=dataset['cdr3'][i]
    return BCR_dict,nt_dict,cdr3_seq_dict
