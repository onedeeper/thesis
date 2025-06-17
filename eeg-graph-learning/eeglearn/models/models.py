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
        self.n_eeg_channels = Config.n_eeg_channels
        self.conv1 = gnn.ChebConv(inchannel, gcn_out_size, K=K)

        self.HC = nn.Sequential(
            nn.Linear(gcn_out_size * self.n_eeg_channels, linear_size),
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
        self.n_eeg_channels = Config.n_eeg_channels
        
        _actual_linear_size_hc = linear_size_hc if linear_size_hc is not None else linear_size
        _actual_drop_rate_hc = drop_rate_hc if drop_rate_hc is not None else drop_rate

        self.conv1 = gnn.ChebConv(inchannel, gcn_out_size, K=K)

        self.HF = nn.Sequential(
            nn.Linear(gcn_out_size * self.n_eeg_channels, linear_size),
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
            nn.Linear(gcn_out_size * self.n_eeg_channels, linear_size),
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
            nn.Linear(gcn_out_size * self.n_eeg_channels, _actual_linear_size_hc),
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
        self.n_eeg_channels = Config.n_eeg_channels
        # inchannel = 5 , which is the number of features
        # for each electrode 
        self.conv1 = gnn.ChebConv(inchannel, gcn_out_size, K=K)
        self.HF = nn.Sequential(
            nn.Linear(gcn_out_size * self.n_eeg_channels, linear_size),
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
            nn.Linear(gcn_out_size * self.n_eeg_channels, linear_size),
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

class EEGNet(nn.Module):
    """Implementation of EEGNet architecture from Lawhern et al. (2018).

    The model consists of 6 sequential blocks:

    1. Temporal Convolution Block
       - Applies F1 temporal filters of size (1, kernel_length)
       - Uses same padding to preserve signal length

    2. Spatial Convolution Block  
       - Applies depthwise convolution with kernel (channels, 1)
       - Uses depth multiplier D to get D*F1 feature maps

    3. First Pooling Block
       - Average pooling with kernel (1,4)
       - Followed by dropout with probability p

    4. Separable Convolution Block
       - Depthwise conv (1,16) with same padding
       - Pointwise 1x1 conv producing F2=D*F1 maps

    5. Second Pooling Block
       - Average pooling with kernel (1,8)
       - Followed by dropout with probability p

    6. Classification Block
       - Flattens features
       - Dense layer mapping to n_classes

    Parameters
    ----------
    n_channels : int
        Number of EEG channels in input
    n_timepoints : int 
        Number of time samples in input
    n_classes : int
        Number of output classes
    F1 : int, default=8
        Number of temporal filters
    D : int, default=2
        Depth multiplier for spatial filters
    kernel_length : int, default=64
        Length of temporal convolution kernel
        Note: Use 32 for data high-passed at ≥4 Hz
    dropout_rate : float, default=0.25
        Dropout probability
        Note: Use 0.25 for cross-subject, 0.5 for within-subject
        
    WRITTEN BY AI
    INSPECTED AND VERIFIED BY AUTHOR
    """

    def __init__(self, n_channels: int, n_timepoints: int, n_classes: int,
                 F1: int = 8, D: int = 2, kernel_length: int = 64,
                 dropout_rate: float = 0.25):
        super().__init__()
        self.F1, self.D, self.F2 = F1, D, F1 * D
        self.dropout_rate = dropout_rate

        # ----- Block 1 -----
        # Temporal convolution (same padding)
        self.conv_temporal = nn.Conv2d(1, F1,
                                       kernel_size=(1, kernel_length),
                                       padding=(0, kernel_length // 2),
                                       bias=False)
        self.bn1 = nn.BatchNorm2d(F1, eps=1e-3, momentum=0.1)

        # Depthwise spatial convolution
        self.conv_spatial = nn.Conv2d(F1, self.F2,
                                      kernel_size=(n_channels, 1),
                                      groups=F1,
                                      bias=False)
        self.bn2 = nn.BatchNorm2d(self.F2, eps=1e-3, momentum=0.1)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # ----- Block 2 (Separable) -----
        # Depthwise
        self.conv_sep_depth = nn.Conv2d(self.F2, self.F2,
                                        kernel_size=(1, 16),
                                        padding=(0, 8),
                                        groups=self.F2,
                                        bias=False)
        # Pointwise
        self.conv_sep_point = nn.Conv2d(self.F2, self.F2,
                                        kernel_size=1,
                                        bias=False)
        self.bn3 = nn.BatchNorm2d(self.F2, eps=1e-3, momentum=0.1)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # ----- Classifier -----
        flat_dim = self._calc_flatten_dim(n_channels, n_timepoints)
        self.classifier = nn.Linear(flat_dim, n_classes)

    # ---------------------------------------------------------------------
    def _calc_flatten_dim(self, C: int, T: int) -> int:
        """Compute dimension of flattened feature map for given input shape."""
        with torch.no_grad():
            x = torch.zeros(1, 1, C, T)
            x = self._forward_features(x)
            return x.numel() // x.size(0)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv_temporal(x)
        x = self.bn1(x)
        x = self.conv_spatial(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv_sep_depth(x)
        x = self.conv_sep_point(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, C, T)`` or ``(batch, 1, C, T)``.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # to (batch, 1, C, T)
        x = self._forward_features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
