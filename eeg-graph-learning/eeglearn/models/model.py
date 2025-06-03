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

class VanillaGraphModel(nn.Module):
    """Joint training model that combines frequency, spatial, and original graph data.
    
    Args:
        inchannel (int): Number of input features per node
        gcn_out_size (int): Number of output features after graph convolution
        batch (int): Batch size
        K (int): Order of Chebyshev polynomials
        linear_size (int): Size of linear layers
        drop_rate (float): Dropout rate
        testmode (bool, optional): If True, 
                            only processes original graph data. Defaults to False
        **kwargs: Additional parameters including:
            - HF (int): Output size for frequency head
            - HS (int): Output size for spatial head
            - HC (int): Output size for classification head for psych labels
    
    Returns:
        tuple: (frequency_output, spatial_output, classification_output) during training
        torch.Tensor: Classification output during testing
    """
    def __init__(self, inchannel, gcn_out_size, batch, K, linear_size, drop_rate,
                 testmode=False, 
                 **kwargs):
        super(VanillaGraphModel, self).__init__()
        self.batch = batch
        self.testmode = testmode
        
        self.conv1 = gnn.ChebConv(inchannel, gcn_out_size, K=K)

        self.HC = nn.Sequential(
            nn.Linear(gcn_out_size * 26, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linear_size, linear_size // 2),
            nn.BatchNorm1d(linear_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linear_size // 2, kwargs['HC'])
        )

    def forward(self, *args):
        if not self.testmode:

            x3, e3 = args[0].x, args[0].edge_index  # original graph data
            x3 = F.relu(self.conv1(x3, e3))
            x3 = x3.view(self.batch, -1)
            x3 = self.HC(x3)
            
            return x3
        else:
            x3, e3 = args[0].x, args[0].edge_index  # original graph data

            x3 = F.relu(self.conv1(x3, e3))
            x3 = x3.view(self.batch, -1)
            logits = self.HC(x3)
            return logits

class JointlyTrainModel(nn.Module):
    """Joint training model that combines frequency, spatial, and original graph data.
    
    Args:
        inchannel (int): Number of input features per node
        gcn_out_size (int): Number of output features after graph convolution
        batch (int): Batch size
        K (int): Order of Chebyshev polynomials
        linear_size (int): Size of linear layers
        drop_rate (float): Dropout rate
        testmode (bool, optional): If True, 
                            only processes original graph data. Defaults to False
        linear_size_hc (int, optional): Size of linear layers for HC head
        drop_rate_hc (float, optional): Dropout rate for HC head
        **kwargs: Additional parameters including:
            - HF (int): Output size for frequency head
            - HS (int): Output size for spatial head
            - HC (int): Output size for classification head for psych labels
    
    Returns:
        tuple: (frequency_output, spatial_output, classification_output) during training
        torch.Tensor: Classification output during testing
    """
    def __init__(self, inchannel, gcn_out_size, batch, K, linear_size, drop_rate,
                 testmode=False, 
                 linear_size_hc=None, drop_rate_hc=None,
                 **kwargs):
        super(JointlyTrainModel, self).__init__()
        self.batch = batch
        self.testmode = testmode
        
        _actual_linear_size_hc = linear_size_hc if linear_size_hc is not None else linear_size
        _actual_drop_rate_hc = drop_rate_hc if drop_rate_hc is not None else drop_rate

        self.conv1 = gnn.ChebConv(inchannel, gcn_out_size, K=K)

        self.HF = nn.Sequential(
            nn.Linear(gcn_out_size * 26, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linear_size, linear_size // 2),
            nn.BatchNorm1d(linear_size//2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linear_size // 2, kwargs['HF'])
        )

        self.HS = nn.Sequential(
            nn.Linear(gcn_out_size * 26, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linear_size, linear_size // 2),
            nn.BatchNorm1d(linear_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linear_size // 2, kwargs['HS'])
        )
        
        self.HC = nn.Sequential(
            nn.Linear(gcn_out_size * 26, _actual_linear_size_hc),
            nn.BatchNorm1d(_actual_linear_size_hc),
            nn.ReLU(inplace=True),
            nn.Dropout(_actual_drop_rate_hc),
            nn.Linear(_actual_linear_size_hc, _actual_linear_size_hc // 2),
            nn.BatchNorm1d(_actual_linear_size_hc // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(_actual_drop_rate_hc),
            nn.Linear(_actual_linear_size_hc // 2, kwargs['HC'])
        )

    def forward(self, *args):
        if not self.testmode:

            x1, e1 = args[0].x, args[0].edge_index  # fre_data
            x2, e2 = args[1].x, args[1].edge_index  # spa_data
            x3, e3 = args[2].x, args[2].edge_index  # original graph data

            x1 = F.relu(self.conv1(x1, e1))
            x2 = F.relu(self.conv1(x2, e2))
            x3 = F.relu(self.conv1(x3, e3))

            x1 = x1.view(self.batch, -1)
            x2 = x2.view(self.batch, -1)
            x3 = x3.view(self.batch, -1)
            

            logits_x1 = self.HF(x1)
            logits_x2 = self.HS(x2)
            logits_x3 = self.HC(x3)
                        

            return logits_x1, logits_x2, logits_x3
        else:
            x3, e3 = args[0].x, args[0].edge_index  # original graph data

            x3 = F.relu(self.conv1(x3, e3))
            x3 = x3.view(self.batch, -1)
            logits_x3 = self.HC(x3)
            return logits_x3

class SelfSupervisedTrain(nn.Module):
    """Self-supervised training model for frequency and spatial graph data.
    
    Args:
        inchannel (int): Number of input features per node
        gcn_out_size (int): Number of output features after graph convolution
        batch (int): Batch size
        K (int): Order of Chebyshev polynomials
        linear_size (int): Size of linear layers
        drop_rate (float): Dropout rate
        **kwargs: Additional parameters including:
            - HF (int): Output size for frequency head (120 permutations)
            - HS (int): Output size for spatial head (128 permutations)
    
    Returns:
        tuple: (frequency_output, spatial_output) with softmax applied
    """
    def __init__(self, inchannel, gcn_out_size, batch, K, linear_size, drop_rate, **kwargs):
        super(SelfSupervisedTrain, self).__init__()
        self.batch = batch

        # inchannel = 5 , which is the number of features
        # for each electrode 
        self.conv1 = gnn.ChebConv(inchannel, gcn_out_size, K=K)

        self.HF = nn.Sequential(
            nn.Linear(gcn_out_size * 26, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linear_size, linear_size // 2),
            nn.BatchNorm1d(linear_size//2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            # This is shape (256 x 120) where 120
            # is the number frequency permutations
            nn.Linear(linear_size // 2, kwargs['HF'])
        )

        self.HS = nn.Sequential(
            nn.Linear(gcn_out_size * 26, linear_size),
            nn.BatchNorm1d(linear_size),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(linear_size, linear_size // 2),
            nn.BatchNorm1d(linear_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            #  this is shape (256 x 128) where 128 is the 
            # number of spatial permutations
            nn.Linear(linear_size // 2, kwargs['HS'])
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

        logits_x1 = self.HF(x1)
        logits_x2 = self.HS(x2)


        return logits_x1, logits_x2

class SelfSupervisedTest(nn.Module):
    """Test model for downstream classification using learned representations.
    
    Args:
        inchannel (int): Number of input features per node
        gcn_out_size (int): Number of output features after graph convolution
        batch (int): Batch size
        K (int): Order of Chebyshev polynomials
        **kwargs: Additional parameters including:
            - classes (int): Number of output classes for classification
    
    Returns:
        torch.Tensor: Classification probabilities with softmax applied
    """
    def __init__(self, inchannel, gcn_out_size, batch, K, **kwargs):
        super(SelfSupervisedTest, self).__init__()
        self.batch = batch

        self.conv1 = gnn.ChebConv(inchannel, gcn_out_size, K=K)

        self.classifier = nn.Sequential(
            nn.Linear(gcn_out_size*26, kwargs['classes'])
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        out = F.relu(self.conv1(x, edge_index))
        out = out.view(self.batch, -1)
        out = self.classifier(out)
        out = F.softmax(out, dim=1)
        return out