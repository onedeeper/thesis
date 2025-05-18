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
K = Config.K
# used for supervised
class JointlyTrainModel(nn.Module):
    def __init__(self, inchannel, outchannel, batch, testmode=False, **kwargs):
        super(JointlyTrainModel, self).__init__()
        self.batch = batch
        self.testmode = testmode
        linearsize = 512


        # K = [1,2,3,4,5,6,7,8,9,10]
        self.conv1 = gnn.ChebConv(inchannel, outchannel, K=K)

        self.HF = nn.Sequential(
            nn.Linear(outchannel * 26, linearsize),
            nn.BatchNorm1d(linearsize),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linearsize, linearsize // 2),
            nn.BatchNorm1d(linearsize//2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
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
            nn.Linear(linearsize // 2, kwargs['HS'])
        )
        
        self.HC = nn.Sequential(
            nn.Linear(outchannel * 26, linearsize),
            nn.BatchNorm1d(linearsize),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linearsize, linearsize // 2),
            nn.BatchNorm1d(linearsize // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linearsize // 2, kwargs['HC'])
        )

        # self.Projection = nn.Sequential(
        #     nn.Linear(outchannel * 26, linearsize),
        #     nn.BatchNorm1d(linearsize),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(drop_rate),
        #     nn.Linear(linearsize, linearsize // 2),
        #     nn.BatchNorm1d(linearsize // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(drop_rate),
        #     nn.Linear(linearsize // 2, linearsize // 4)
        # )


    def forward(self, *args):
        if not self.testmode:

            x1, e1 = args[0].x, args[0].edge_index  # fre_data
            x2, e2 = args[1].x, args[1].edge_index  # spa_data
            x3, e3 = args[2].x, args[2].edge_index  # original graph data
            # x4, e4 = args[3].x, args[3].edge_index  # contrastive learning data
            # x5, e5 = args[4].x, args[4].edge_index  # contrastive learning data

            # x1 = torch.tensor(x1, dtype=torch.float32)
            # x1 = x1.float()

            x1 = F.relu(self.conv1(x1, e1))
            x2 = F.relu(self.conv1(x2, e2))
            x3 = F.relu(self.conv1(x3, e3))
            # x4 = F.relu(self.conv1(x4, e4))
            # x5 = F.relu(self.conv1(x5, e5))

            x1 = x1.view(self.batch, -1)
            x2 = x2.view(self.batch, -1)
            x3 = x3.view(self.batch, -1)
            # x4 = x4.view(self.batch, -1)
            # x5 = x5.view(self.batch, -1)

            x1 = self.HF(x1)
            x2 = self.HS(x2)
            x3 = self.HC(x3)
            # x4 = self.Projection(x4)
            # x5 = self.Projection(x5)

            x1 = F.softmax(x1, dim=1)
            x2 = F.softmax(x2, dim=1)
            x3 = F.softmax(x3, dim=1)
            # x4 = F.normalize(x4, dim=-1)
            # x5 = F.normalize(x5, dim=-1)

            return x1, x2, x3
        else:
            x3, e3 = args[0].x, args[0].edge_index  # original graph data

            x3 = F.relu(self.conv1(x3, e3))
            x3 = x3.view(self.batch, -1)
            x3 = self.HC(x3)
            x3 = F.softmax(x3, dim=1)
            return x3

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