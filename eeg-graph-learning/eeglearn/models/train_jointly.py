import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from eeglearn.config import Config
from eeglearn.preprocess.preprocessing import Preproccesing
from eeglearn.utils.utils import get_details_from_file_name, get_cleaned_data_paths,\
                            load_preprocessed_data
from operator import itemgetter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from multiprocessing import cpu_count

from microstructpy.geometry import Ellipsoid
from pygeodesy.ellipsoidalVincenty import Cartesian
from pygeodesy import Ellipsoid as pyg_Ellipsoid
from pygeodesy.ellipsoidalKarney import LatLon as KLatLon
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from AutoWeight import AutomaticWeightedLoss
from eeglearn.models.model import SelfSupervisedTrain
from eeglearn.features.graphs import Graphs
import json
import pandas as pd

"""Self-supervised  + multi-task learning EEG training pipeline .

Implementation of a self-supervised training approach for EEG data based on Li et al. 2023
(https://ieeexplore.ieee.org/abstract/document/9765326). This module handles data splitting,
model training, and metrics tracking for both frequency and spatial graph representations.

Functions:
    split_data: Split participants into train/test/validation sets
    train: Execute the self-supervised training process and save metrics
"""