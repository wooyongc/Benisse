from __future__ import print_function
import torch.nn as nn


class MyAlexNetCMC(nn.Module):
    def __init__(self, in_feature=130, feat_dim=15, freeze=False):
        super(MyAlexNetCMC, self).__init__()
        self.encoder = alexnet(feat_dim=feat_dim,in_feature=in_feature)
        self.encoder = nn.DataParallel(self.encoder)
        if freeze:
            for parameters in self.encoder.parameters():
                parameters.requires_grad_(False)

    def forward(self, x, layer=5):
        return self.encoder(x, layer)

class alexnet(nn.Module):
    def __init__(self, feat_dim=15,in_feature=130):
        super(alexnet, self).__init__()

        self.l_to_ab = alexnet_cdr_to_vdj(in_channel=1, feat_dim=feat_dim)
        self.ab_to_l = alexnet_vdj_to_cdr(in_feature=in_feature, feat_dim=feat_dim)

    def forward(self, data, layer=8):
        l, ab,cdr3_seq = data['cdr'], data['vdj'], data['cdr3_seq']
        feat_l = self.l_to_ab(l, layer)
        feat_ab = self.ab_to_l(ab, layer)
        return feat_l, feat_ab, cdr3_seq

class alexnet_cdr_to_vdj(nn.Module):
    def __init__(self, in_channel=1, feat_dim=15):
        super(alexnet_cdr_to_vdj, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channel, 6, kernel_size=(1,3), stride=1, padding=(0,1),bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), padding=(0,0)),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(6, 6,kernel_size=(5,4),stride=1,padding=(0,1),bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), padding=(0,0)),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(6, 12,kernel_size=(1,2),stride=1,padding=(0,0),bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2), padding=(0,0)),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(48,100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(100, feat_dim)
        )
    def forward(self, x, layer=6):
        if layer <= 0:
            return x
        x = self.conv_block_1(x)
        if layer == 1:
            return x
        x = self.conv_block_2(x)
        if layer == 2:
            return x
        x = self.conv_block_3(x)
        x = x.view(x.size()[0],-1)
        if layer == 3:
            return x
        x = self.fc4(x)
        if layer == 4:
            return x
        x = self.fc5(x)
        return x

class alexnet_vdj_to_cdr(nn.Module):
    def __init__(self, in_feature=130, feat_dim=15):
        super(alexnet_vdj_to_cdr, self).__init__()
     
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=(3), stride=1, padding=2,bias=False),
            nn.BatchNorm1d(6),
            nn.ReLU(inplace=True),
        ) 
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(6, 6, kernel_size=(3), stride=1, padding=0,bias=False),
            nn.BatchNorm1d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, padding=0),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv1d(6, 12,kernel_size=(3),stride=1,padding=0,bias=False),
            nn.BatchNorm1d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, padding=0),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(156, 80),
            nn.BatchNorm1d(80),
            nn.ReLU(inplace=True),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(80, feat_dim)
        )
       
    def forward(self, x, layer=5):
        x=x-0.5
        if layer <= 0:
            return x
        x=x.unsqueeze_(1)
        x = self.conv_block_1(x)
        if layer == 1:
            return x
        x = self.conv_block_2(x)
        if layer == 2:
            return x
        x = self.conv_block_3(x)
        if layer == 3:
            return x
        x = x.contiguous().view(x.size()[0],-1)
        x = self.fc4(x)
        if layer == 4:
            return x
        x = self.fc5(x)
        return x
