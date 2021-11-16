import torch.nn as nn
import torch.nn.functional as F

class my_focal_loss(nn.Module):
    def __init__(self, alpha=0.90,gamma=2.0, reduction='mean'):
        super(my_focal_loss, self).__init__()
        self.alpha=alpha
        self.gamma=gamma
        self.reduction = reduction
    def forward(self,pred,target):
        pt=(1-pred)*target+pred*(1-target)
        focal_weight=pt.pow(self.gamma)*(self.alpha*target+(1-self.alpha)*(1-target))
        loss=F.binary_cross_entropy(pred,target,reduction='none')*focal_weight
        assert self.reduction=='mean'

        #return loss.mean()
        return loss.sum()