"""A class to define the ML models.

Recreates the models in Li. et al 2023 for the Td-brain dataset.
(https://ieeexplore.ieee.org/abstract/document/9765326)

Created on: May 2025
Author: Udesh Habaraduwa

Attributes
----------

Methods
-------
"""

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from eeglearn.config import Config

drop_rate = Config.drop_rate

class SelfSupervisedTrain(nn.Module):
    def __init__(self, inchannel, outchannel, batch, **kwargs):
        super(SelfSupervisedTrain, self).__init__()
        self.batch = batch

        linearsize = 512

        # inchannel = 5 , which is the number of features
        # for each electrode 
        self.conv1 = gnn.ChebConv(inchannel, outchannel, K=2)

        self.HF = nn.Sequential(
            nn.Linear(outchannel * 26, linearsize),
            nn.BatchNorm1d(linearsize),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linearsize, linearsize // 2),
            nn.BatchNorm1d(linearsize//2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            # This is shape (256 x 120) where 120
            # is the number frequency permutations
            nn.Linear(linearsize // 2, kwargs['HF'])
        )

        self.HS = nn.Sequential(
            nn.Linear(outchannel * 26, linearsize),
            nn.BatchNorm1d(linearsize),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linearsize, linearsize // 2),
            nn.BatchNorm1d(linearsize // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            #  this is shape (256 x 128) where 128 is the 
            # number of spatial permutations
            nn.Linear(linearsize // 2, kwargs['HS'])
        )

    def forward(self, *args):

        # in this case, x is of shape (26 x 5)
        x1, e1 = args[0].x, args[0].edge_index  # fre_data
        x2, e2 = args[1].x, args[1].edge_index  # spa_data

        # feature extraction by updating the node features with a GCN
        x1 = F.relu(self.conv1(x1, e1))
        x2 = F.relu(self.conv1(x2, e2))
    
        # change them into a shape for the NN layers 
        # which expects a batch of data.
        x1 = x1.view(self.batch, -1)
        x2 = x2.view(self.batch, -1)

        x1 = self.HF(x1)
        x2 = self.HS(x2)

        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)

        return x1, x2

class SelfSupervisedTest(nn.Module):
    def __init__(self, inchannel, outchannel, batch, **kwargs):
        super(SelfSupervisedTest, self).__init__()
        self.batch = batch

        linearsize = 512

        self.conv1 = gnn.ChebConv(inchannel, outchannel, K=2)

        self.classifier = nn.Sequential(
            nn.Linear(outchannel*26, kwargs['classes'])
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        out = F.relu(self.conv1(x, edge_index))
        out = out.view(self.batch, -1)
        out = self.classifier(out)
        out = F.softmax(out, dim=1)
        return out