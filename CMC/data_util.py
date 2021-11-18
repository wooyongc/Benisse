import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    __initialized = False
    def __init__(self, indexes, cdr_dict, vdj_dict,cdr3_seq_dict):
        """
        Args:
            indexes: index (keys) used for the three dicts
        """
        self.indexes = indexes
        self.indexes_dict = dict()
        for i, _ in enumerate(indexes):
            self.indexes_dict[_] = i
        self.cdr_dict = cdr_dict
        self.vdj_dict = vdj_dict
        self.cdr3_seq_dict = cdr3_seq_dict
        self.__initialized = True

    def __len__(self):
        """Denotes the number of samples"""
        return len(self.indexes)
    
    def __getitem__(self, index):
        """Generate one batch of data.
        Returns:
            The __getitem__ method takes multiple keys and returns __data_generation data.
        """
        # Generate torch.long indexes of the batch samples
        data_index=self.indexes[index]
        idx=index
        # Generate data
        data = self.__data_generation(data_index)
        return data, idx
    
    def __data_generation(self, indexes):
        """Generates data containing batch_size samples.
        Returns:
            data: a dictionary with data.cdr in b*encode_dim*5; data.vdj in b*len(VDJlabelvector)
        """
        cdr=self.cdr_dict[indexes]
        vdj = self.vdj_dict[indexes]
        cdr3_seq = self.cdr3_seq_dict[indexes]
        data = dict()
        cdr=np.transpose(cdr)
        data['cdr']=torch.FloatTensor(cdr[np.newaxis,:])#tensor shape: (batch,channel,height,width), add channel =1
        data['vdj']=vdj.squeeze()
        data['cdr3_seq']=cdr3_seq
        return data
