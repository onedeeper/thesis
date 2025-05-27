"""Self-supervised EEG training pipeline.

Written by Li et al. 
Copied from  Li et al. 2023(https://ieeexplore.ieee.org/abstract/document/9765326). 
Modified by Udesh Habaraduwa, May 2025

"""

import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    """Automatically weighted multi-task loss function that
    learns optimal weights for each task.
    
    Args:
        num (int): Number of loss terms to combine
        
    Returns:
        torch.Tensor: Weighted sum of all loss terms
        
    Example:
        >>> loss1 = torch.tensor(1.0)
        >>> loss2 = torch.tensor(2.0)
        >>> awl = AutomaticWeightedLoss(2)
        >>> total_loss = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        #print(self.params)

    def forward(self, *x):
        loss_sum = 0
        length = len(x)-1
        for i, loss in enumerate(x):
            if i == length:
                loss_sum += 1 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
            else:
                loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
        return loss_sum

if __name__ == '__main__':
    awl = AutomaticWeightedLoss(2)
    print(awl.parameters())
