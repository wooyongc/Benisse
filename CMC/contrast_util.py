"""NCEAverage, serve as contrast
"""
import math
import torch
from torch import nn
from alias_multinomial import AliasMethod

class NCEAverage(nn.Module):

    def __init__(self, input_size, output_size, K, T=0.07, momentum=0.5):
        """
        Args:
            input_size: n_features
            output_size: n_samples
            K: number of negatives to constrast for each positive
            T: temperature that modulates the distribution
        """
        super(NCEAverage, self).__init__()
        self.output_size = output_size
        self.unigrams = torch.ones(self.output_size)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K
        self.register_buffer('params', torch.FloatTensor([K, T, momentum]))
        stdv = 1. / math.sqrt(input_size / 3)
        self.register_buffer('memory_cdr', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_vdj', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))

    def forward(self, cdr, vdj, index, idx=None):
        """
        Args:
            cdr: out_features for cdr
            vdj: out_features for vdj
            index: torch.long for data ids
        """
        K = int(self.params[0].item())
        T = self.params[1].item()
        momentum = self.params[2].item()
        batch_size = cdr.size(0)
        output_size = self.memory_cdr.size(0)
        input_size = self.memory_cdr.size(1)
        # score computation
        if idx is None:
            idx = self.multinomial.draw(batch_size * (self.K + 1)).view(batch_size, -1)
            idx.select(1, 0).copy_(index.data)
        # sample
        weight_vdj = torch.index_select(self.memory_vdj, 0, idx.view(-1)).detach()
        weight_vdj = weight_vdj.view(batch_size, K + 1, input_size)
        out_cdr = torch.bmm(weight_vdj, cdr.view(batch_size, input_size, 1))
        # sample
        weight_cdr = torch.index_select(self.memory_cdr, 0, idx.view(-1)).detach()
        weight_cdr = weight_cdr.view(batch_size, K + 1, input_size)
        out_vdj = torch.bmm(weight_cdr, vdj.view(batch_size, input_size, 1))
        out_cdr = torch.div(out_cdr, T)
        out_vdj = torch.div(out_vdj, T)
        out_cdr = out_cdr.contiguous()
        out_vdj = out_vdj.contiguous()
        # # update memory
        with torch.no_grad():
            cdr_pos = torch.index_select(self.memory_cdr, 0, index.view(-1))
            cdr_pos.mul_(momentum)
            cdr_pos.add_(torch.mul(cdr, 1 - momentum))
            cdr_norm = cdr_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_cdr = cdr_pos.div(cdr_norm)
            self.memory_cdr.index_copy_(0, index, updated_cdr)

            vdj_pos = torch.index_select(self.memory_vdj, 0, index.view(-1))
            vdj_pos.mul_(momentum)
            vdj_pos.add_(torch.mul(vdj, 1 - momentum))
            vdj_norm = vdj_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_vdj = vdj_pos.div(vdj_norm)
            self.memory_vdj.index_copy_(0, index, updated_vdj)

        return out_cdr, out_vdj
    
# Loss functions, adapted from the CMC github
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss"""
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        #super spicified that only call the init function from NCESoftmasLoss
        #rather than nn.Module.
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        
        return loss
