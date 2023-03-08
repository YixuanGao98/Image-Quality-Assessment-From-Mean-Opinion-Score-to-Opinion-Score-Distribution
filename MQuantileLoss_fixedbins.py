import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy


class MQuantileLoss(nn.Module):
    def __init__(self):
        super(MQuantileLoss, self).__init__()

    def forward(self, p_estimate: Variable, p_target: Variable):
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        percentiles =[0.25, 0.5,0.75]
        # x= torch.arange(5, 105, 10)
        x= torch.arange(1, len(p_target[0])+1, 1)
        quan1=torch.zeros([len(p_target),len(percentiles)]).cuda()
        quan2=torch.zeros([len(p_estimate),len(percentiles)]).cuda()
        for index,target in enumerate(cdf_target):
            for k in range(0, len(percentiles)):
                for i in range(0, len(target)):
                    if percentiles[k] <= target[0]:
                        Xa = 0
                        Xb = x[i]
                        Ya = 0
                        Yb = target[i]

                        A = (Yb-Ya)/(Xb-Xa)
                        B = Yb-A*Xb

                        score1 = (percentiles[k]-B)/A
                        break
                    elif percentiles[k] <= target[i] and percentiles[k] > target[i-1]:
                        Xa = x[i-1]
                        Xb = x[i]
                        Ya = target[i-1]
                        Yb = target[i]

                        A = (Yb-Ya)/(Xb-Xa)
                        B = Yb-A*Xb

                        score1 = (percentiles[k]-B)/A
                        break
                quan1[index][k]=score1
                
        for index,pre0 in enumerate(cdf_estimate):
            for k in range(0, len(percentiles)):
                for i in range(0, len(pre0)):
                    if i == 0 and percentiles[k] <= pre0[i]:
                        Xa = 0
                        Xb = x[i]
                        Ya = 0
                        Yb = pre0[i]

                        A = (Yb-Ya)/(Xb-Xa)
                        B = Yb-A*Xb 

                        score2 = (percentiles[k]-B)/A
                        break
                    elif percentiles[k]<= pre0[i] and percentiles[k] > pre0[i-1]:
                        Xa = x[i-1]
                        Xb = x[i]
                        Ya = pre0[i-1]
                        Yb = pre0[i]

                        A = (Yb-Ya)/(Xb-Xa)
                        B = Yb-A*Xb

                        score2 = (percentiles[k]-B)/A
                        break
                quan2[index][k]=score2
        loss=torch.abs(quan1-quan2)
        



        

        return loss.mean()