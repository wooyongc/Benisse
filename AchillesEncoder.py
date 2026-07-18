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
    parser.add_argument('--output_data',type=str,help='Encoded BCR sequences, the output data file')
    parser.add_argument('--encode_dim',type=int,default=40,help='Columns of padded atchley matrix (default: 40)')
    parser.add_argument('--pad_length',type=int,default=130,help='Length of padded nucleotide sequence (default: 130), \
        cells with nucleotide sequences longer than the length will be removed.')
    parser.add_argument('--cuda', default=True, type=strtobool, help='Whether to use CUDA to train model')
    opt=parser.parse_args()
    return opt

#Calculate 
def embed(test_loader, model, device):

    feature_array = []

    for idx, (data, index) in enumerate(test_loader):
        index = index.to(device)
        for _ in list(data.keys())[0:2]:
            data[_] = data[_].float().to(device)
        feat_cdr, feat_vdj, cdr3_seq = model(data)
        feature_array_tmp = pd.DataFrame(feat_cdr.cpu().numpy())
        feature_array_tmp['index'] = cdr3_seq
        feature_array.append(feature_array_tmp)

    feature_array = pd.concat(feature_array)
    feature_array.sort_values(by=['index'], inplace=True, kind='mergesort')

    return feature_array

def encode_bcr(input_data, output_data, encode_dim=40, pad_length=130, cuda=True):
    """Encode BCR CDR3 sequences with the frozen Benisse reference model.

    This is the internal callable boundary used by parity tests and future data
    adapters. The legacy command-line interface delegates here unchanged.
    """
    opt = argparse.Namespace(
        input_data=str(input_data),
        output_data=str(output_data),
        encode_dim=encode_dim,
        pad_length=pad_length,
        cuda=cuda,
    )

    #Load data
    test = load_BCRdata2(opt)

    #Data prep
    aa_dict = dict()
    with open(os.path.dirname(os.path.abspath(__file__)) + "/dependency/Atchley_factors.csv",'r') as aa:
        aa_reader = csv.reader(aa)
        next(aa_reader, None)
        for rows in aa_reader:
            aa_name = rows[0]
            aa_factor = rows[1:len(rows)]
            aa_dict[aa_name] = np.asarray(aa_factor, dtype='float')
    
    cdr_test, vdj_test, cdr3_seq_test = datasetMap_nt(test, aa_dict, opt.encode_dim, opt.pad_length)
    batch_size = 64
    random_seed = 123
    # Reproducibility: seed RNGs and pin CPU threads so encoding is bit-deterministic.
    # (Multi-threaded reduction order and hash-randomized set() ordering otherwise cause
    #  ~1e-6 run-to-run drift, which propagates into the R stage.)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.set_num_threads(1)
    # sorted() (not list(set(...))) makes iteration order independent of PYTHONHASHSEED
    test_indices = sorted(set(vdj_test.keys()))
    cdr_shape = cdr_test[list(cdr_test.keys())[0]].shape[0]
    n_vdj = vdj_test[list(vdj_test.keys())[0]].size()[0]
    n_data = len(test_indices)
    n_out_features = 20

    test_set = Dataset(test_indices, cdr_test, vdj_test, cdr3_seq_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, 
                                              shuffle=False, sampler=None, batch_sampler=None, num_workers=0)

    #Load model
    if torch.cuda.is_available():
        if opt.cuda:
            device = torch.device("cuda")
        if not opt.cuda:
            print("Note: You have CUDA enabled but not using it.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    feat_dim = 20
    in_feature = 130
    nce_k = 1
    nce_t = 0.2
    nce_m = 0.9
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.0001
    gradient_clip = 5

    print("Loading trained model...")

    # Load onto the resolved device: map_location=device works for both CPU and CUDA,
    # and avoids crashing when --cuda True is passed on a machine without a CUDA GPU.
    state = torch.load(os.path.dirname(os.path.abspath(__file__)) + "/dependency/trained_model.pt", map_location=device)

    test_model = MyAlexNetCMC(in_feature=in_feature, feat_dim=feat_dim, freeze=True).to(device)
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

    print("Embedding BCR sequences...")
    encoded_BCR = embed(test_loader, test_model, device)
    encoded_BCR.to_csv(opt.output_data, sep=',')
    print("BCR successfully encoded. The output file is generated here: ", opt.output_data)
    return encoded_BCR


def main():
    opt = parse_option()
    encode_bcr(
        input_data=opt.input_data,
        output_data=opt.output_data,
        encode_dim=opt.encode_dim,
        pad_length=opt.pad_length,
        cuda=opt.cuda,
    )


if __name__ == '__main__':
    main()

# Ze, we keep you in our heart forever
