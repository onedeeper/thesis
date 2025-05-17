import os
from pathlib import Path

import numpy as np
import torch
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

"""A class to run a self-supervised training pipeline.

Created on: May 2025
Author: Udesh Habaraduwa

Attributes
----------

Methods
-------
"""