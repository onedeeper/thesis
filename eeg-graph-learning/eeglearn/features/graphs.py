"""A class to generate graph representations of EEG energy band feature data.

Created on: May 2025
Author: Udesh Habaraduwa

Attributes
----------


Methods
-------

"""
import os
import pickle
import random
import shutil
import tempfile
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
import torch

from eeglearn.config import Config
from eeglearn.features.energy import Energy
from eeglearn.preprocess.preprocessing import Preproccesing
from eeglearn.utils.utils import get_participant_id_condition_from_string, hamming_set,\
    get_cleaned_data_paths, load_preprocessed_data
from operator import itemgetter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os 

from microstructpy.geometry import Ellipsoid
from pygeodesy.ellipsoidalVincenty import LatLon, Cartesian
from pygeodesy import Datum
from pygeodesy import Ellipsoid as pyg_Ellipsoid
from pygeodesy.ellipsoidalKarney import LatLon as KLatLon
class Graphs():
    """A container for graph representation generation.

    Methods
    -------
    __init__:
        Initialize the Graphs class.
    get_graphs:
        Generate a dataloader which provides graph representations of a given dataset.
    get_adjacency:
        Compute the adjacency matrix for a given example.
    """
    def __init__(self, distance : str,
                 cleaned_data_path : str, 
                 batch_size : int = 256) -> None:
        """Generate the graph representation for a the data from participants.

        Args:
        ----
            cleaned_data_path : Path to pre-processed objects (not features)
            distance : str : "eucledian" | "great_circle" | "ellipsoid"
                       The distance metric to use when calculating the nearest 
                       neighbors for adjacency.
            batch_size : Size of mini batches to divide the dataset into.

        Returns:
        ----
            None
        """
        dist_types = ["eucledian", "great_circle", "ellipsoid"]
        assert distance in dist_types,\
            "Must be one of eucledian, great_circle, or ellipsoid"
        self.dist_type  : str = distance
        self.batch_size : int = batch_size
        self.ch_positions : dict = {'Fp1' : np.array([-0.02681, 0.08406, -0.01056]),
        'Fp2' : np.array([0.02941, 0.08374, -0.01004]),
        'F7'  : np.array([-0.06699, 0.04169, -0.01596]),
        'F3'  : np.array([-0.04805, 0.05187, 0.03987]),
        'Fz'  : np.array([0.00090, 0.05701, 0.06636]),
        'F4'  : np.array([0.05038, 0.05184, 0.04133]),
        'F8'  : np.array([0.06871, 0.04116, -0.01531]),
        'FC3' : np.array([-0.05883, 0.02102, 0.05482]),
        'FCz' : np.array([0.00057, 0.02463, 0.08763]),
        'FC4' : np.array([0.06029, 0.02116, 0.05558]), 
        'T7'  : np.array([-0.08336, -0.01652, -0.01265]), 
        'C3'  : np.array([-0.06557, -0.01325, 0.06498]),
        'Cz'  : np.array([0.000023, -0.01128, 0.09981]),
        'C4'  : np.array([0.06650, -0.01280, 0.06511]),
        'T8'  : np.array([0.08444, -0.01665, -0.01179]), 
        'CP3' : np.array([-0.06551, -0.04848, 0.06857]),
        'CPz' : np.array([-0.0042, -0.04877, 0.09837]), 
        'CP4' : np.array([0.06503, -0.04835, 0.06857]), 
        'P7'  : np.array([-0.07146, -0.07517, -0.00370]), 
        'P3'  : np.array([-0.05507, -0.08011, 0.05944]), 
        'Pz'  : np.array([-0.00087, -0.08223, 0.08243]),
        'P4'  : np.array([0.05351, -0.08013, 0.05940]), 
        'P8'  : np.array([0.07110, -0.07517, -0.00369]), 
        'O1'  : np.array([-0.02898, -0.11452, 0.00967]),  
        'Oz'  : np.array([-0.00141, -0.11779, 0.01584]),
        'O2'  : np.array([0.02689, -0.11468, 0.00945])
        }
        self.td_brain_channels : list[str] = [  'Fp1', 'Fp2', 'F7', 'F3', 
                                                'Fz', 'F4', 'F8', 'FC3', 
                                                'FCz', 'FC4', 'T7', 'C3', 
                                                'Cz', 'C4', 'T8', 'CP3',
                                                'CPz', 'CP4', 'P7', 'P3', 
                                                'Pz', 'P4', 'P8', 'O1',
                                                'Oz', 'O2']
        self.cleaned_data_path : str = cleaned_data_path

    def get_graphs(self,data : list[tuple[torch.Tensor,
                                          np.ndarray,
                                          str]]):
        
        dataset = [ participant[0] for participant in data]
        #print(len(dataset))
        return DataLoader(dataset, batch_size = self.batch_size)
    
    def get_bad_channels(self) -> None :
        """Retreive the bad channels from preprocessed data."""
        participant_list = os.listdir(self.cleaned_data_path)
        folders_and_files, participant_npy_files = \
            get_cleaned_data_paths(participant_list=participant_list,
                                   cleaned_path=self.cleaned_data_path)
        
        bad_channels : dict[str, list[str]] = {}
        for file in folders_and_files:
            folder_path : str = file[0]
            preprocessed_file_name : str = file[1]
            prep_data = load_preprocessed_data(folder_path=folder_path,
                                   file_name=preprocessed_file_name)
            bad= prep_data.bad_channels_after_interpolation['bad_all']
            bad_channels[preprocessed_file_name] = bad
        return bad_channels

    def get_distance(self) -> np.ndarray:
        """Calculates the node distance"""
        if self.dist_type == "ellipsoid":
            return self.get_ellipsoid_distance()
        else:
            raise NotImplementedError("Eucledian and great circle to be impelemtended")
    
    def get_ellipsoid_distance(self) -> np.ndarray:
        """Calculates the ellipsoidal distance between electrodes.
        
        Model the head as an ellipsoid and calculate the distance between electrodes
        as ellipsoid geodesic length. To avoid the problem discussed in [1]  using the
        Vincenty algorithm, the Karney algorithm is used to calcualte the distance.

        reference : [1] (https://ieeexplore.ieee.org/abstract/document/7833851)
        """
        e = Ellipsoid()
        # Fit an ellipsoid to the position of the EEG electrodes
        fit_geo = e.best_fit(points= [values.tolist() for values in 
                                      self.ch_positions.values()])

        # center : center of the fitted ellipsoid
        # axes : The radii lengths of the 3 semi-axes
        # rot_seq : How the fitted ellipsoid is rotated in space
        (center, axes, rot) = (
            fit_geo.center,
            fit_geo.axes,
            fit_geo.rot_seq
        )

        # Build a spheroid with the 3 axes
        r = np.sort(axes)
        # Equitorial (x,y) radius is the mean of the two largest.
        # Polar is the shortest
        equitorial_rad, polar_rad = float(r[-2:].mean()), float(r[0])
        head = pyg_Ellipsoid(a= equitorial_rad, b= polar_rad)
        dists = []
        for ch_i in self.td_brain_channels:
            row = []
            p1 = Cartesian(*self.ch_positions[ch_i], datum = head).toLatLon()
            for ch_j in self.td_brain_channels:
                p2 = Cartesian(*self.ch_positions[ch_j], datum = head).toLatLon()       
                p1k = KLatLon(p1.lat, p1.lon, datum=p1.datum)
                p2k = KLatLon(p2.lat, p2.lon, datum=p2.datum)
                d   = p1k.distanceTo(p2k)  
                row.append(d)
            dists.append(row)

        return np.array(dists)
    
    def get_adjacency(self) -> None:
        """Calculate the adajcency matrix for each participant."""
        
        return None