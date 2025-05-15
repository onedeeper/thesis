"""Tests for the Graphs module.

This module contains test cases for the Energy class and its functionality including:
- Graphs initialization and configuration
- Graph representation generation from EEG
- Adajacency matrix generation
- Parallel processing capabilities
"""

import os
import pickle
import random
import shutil
import tempfile
import random
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
import torch
import tempfile
from eeglearn.config import Config
from eeglearn.features.energy import Energy
from eeglearn.preprocess.preprocessing import Preproccesing
from eeglearn.features.graphs import Graphs
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from eeglearn.utils.utils import get_details_from_file_name, hamming_set,\
    get_cleaned_data_paths
from operator import itemgetter
#Config.RANDOM_SEED = 1223333

from microstructpy.geometry import Ellipsoid
from pygeodesy.ellipsoidalVincenty import LatLon, Cartesian
from pygeodesy import Datum
from pygeodesy import Ellipsoid as pyg_Ellipsoid

Config.RANDOM_SEED = 212112
Config.set_global_seed()

class TestGraphs:
    """Test class for evalutating functionality of the Graphs module."""

    @pytest.fixture(autouse= True)
    def setup(self):
        self.cleaned_data_path = \
            os.environ.get("EEG_TEST_CLEANED_FOLDER_PATH")
        n_test_files : int = 5
        # this would be 128 for spatial permutations
        n_max_test_labels : int = 120
        n_epochs : int = np.random.randint(1,12)
        n_perms_per_epoch : int = np.random.randint(2,5)
        n_channels :int = np.random.randint(1,26)
        n_bands : int = np.random.randint(1,5)
        subj_ids : list[int] = [np.random.randint(10000000,15000000)\
                                for _ in range(n_test_files)]
        
        conditions : list[str] = ["EO", "EC"]
        test_file_ids =\
            [f"energy_sub-{subj_id}_{random.sample(conditions,1)}_epoched.pt"\
                              for subj_id in subj_ids]
        
        self.td_brain_channels : list[str] = [  'Fp1', 'Fp2', 'F7', 'F3', 
                                                'Fz', 'F4', 'F8', 'FC3', 
                                                'FCz', 'FC4', 'T7', 'C3', 
                                                'Cz', 'C4', 'T8', 'CP3',
                                                'CPz', 'CP4', 'P7', 'P3', 
                                                'Pz', 'P4', 'P8', 'O1',
                                                'Oz', 'O2']
        self.test_file_list = [(
            torch.rand((n_epochs,n_perms_per_epoch,n_channels,n_bands)), # data
            np.random.randint(0,n_max_test_labels, 
                              size = (n_epochs, n_perms_per_epoch)), # labels per epoch
            file_id)
              for file_id in test_file_ids]
        
    def test_graph_init(self) -> None:
        """Test creation of Graphs object.
        
        Args:
        ----

        Returns:
        ----
            None
        """
        graphs = Graphs(distance="ellipsoid",
                        cleaned_data_path=self.cleaned_data_path)
        assert(isinstance(graphs, Graphs))

        # should fail with strings that are not as intended
        with pytest.raises(Exception):
            dist = "sfe323r32asfe"
            graphs = Graphs(distance=dist,
                            cleaned_data_path=self.cleaned_data_path)
        with pytest.raises(NotImplementedError):
            graphs = Graphs(distance="eucledian",
                            cleaned_data_path=self.cleaned_data_path)
    def test_get_graphs(self) -> None:
        """Test the generation of graph objects.
        
        Args:
        ----

        Returns:
        ----
            None
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # create some temporary preprocessed objects with bad channels 
            # Make that the cleaned data path. 
            # Make these file ids the same as the test data generated here.
            # created preprocessing objects need to have the same file names
            # as the files here.
            for data,_,file_id in self.test_file_list:
                bads: list[str] = random.sample(self.td_brain_channels,4)
                temp_file, temp_dir_cleaned= self.create_temp_preprocessed_object(
                                                    temp_dir=temp_dir,
                                                            bads=bads,
                                                            file_id=file_id)
                save_path : Path = Path(temp_dir) / "perms"
                save_path.mkdir(parents=True,exist_ok = True)
                with open(save_path / file_id , 'wb') as output:   
                    pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)
                assert os.path.exists(save_path/file_id)
            graphs = Graphs(distance="ellipsoid",
                                cleaned_data_path=temp_dir_cleaned)
            graph_loader = graphs.get_graphs(perms_path=save_path,
                                            files_to_load=\
                                                [id for _,_,id in 
                                                 self.test_file_list[:2]])
            # want to test of the bads are handled properly
            # I can pull out the specific rows and columns and check if they are all 0

            assert isinstance(graph_loader, DataLoader),\
                "Should return torch geometric dataloader"
        
            # check the contents of a batch
            graph_loader_iter = iter(graph_loader)
            first_example = next(graph_loader_iter)[0]

            # each example should be a graph data object
            #assert isinstance(first_example, Data)

    def test_get_ellipsoid_distance(self) -> None:
        """Test the calculation of ellipsoidal distances between electrodes.
        
        Args:
        ----

        Returns:
        ----
            None
        """
        graphs = Graphs(distance="ellipsoid",
                        cleaned_data_path=self.cleaned_data_path)
        distance = graphs.get_ellipsoid_distance()
        assert isinstance(distance, np.ndarray), "Expected an array"
        assert distance.shape == (26,26), "Should contain distance between each node"
        assert np.allclose(np.diag(distance), np.zeros(26)), "Distance to self == 0"
        assert np.allclose(distance - distance.T, 0), "Should be symmetric"
        assert np.min(distance) >= 0, "Distances should be strictly greater than 0"

    def test_get_distance(self) -> None:
        """Test the calculation of distances between electrodes by chosen metric.
        
        Args:
        ----

        Returns:
        ----
            None
        """
        graphs = Graphs(distance="ellipsoid",
                        cleaned_data_path=self.cleaned_data_path)
        distance = graphs.get_distance()
        assert isinstance(distance, np.ndarray), "Expected an array"
        assert distance.shape == (26,26), "Should contain distance between each node"
        assert np.allclose(np.diag(distance), np.zeros(26)), "Distance to self == 0"
        assert np.allclose(distance - distance.T, 0), "Should be symmetric"
        assert np.min(distance) >= 0, "Distances should be strictly greater than 0"

    def test_get_bads(self) -> None:
        """Test the loading of bad channels for a given participant and sesion.
        Args:
        ----

        Returns:
        ----
            None
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            bads: list[str] = ["Fp1", "P3"]
            temp_file, temp_dir = self.create_temp_preprocessed_object(
                                                    temp_dir=temp_dir,
                                                            bads=bads)
            graphs = Graphs(distance="ellipsoid",
                    cleaned_data_path=temp_dir)
            bads_found = graphs.get_bad_channels()

            #bad_indices = [graphs.td_brain_channels.index(bad) for bad in bads]
            assert bads_found[temp_file] == bads,"Should correctly load the forced bads"
    
    def test_get_adjacency(self)-> None:
        """Test the generation of adjacency matrix accounting for bad channels
        
        Args:
        ----
            
        Returns:
        ----
            None
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            bads: list[str] = ["Fp1", "Oz"]
            temp_file, temp_dir = self.create_temp_preprocessed_object(
                                                    temp_dir=temp_dir,
                                                            bads=bads)
            graphs = Graphs(distance="ellipsoid",
                    cleaned_data_path=temp_dir)
            row_idxs, col_idxs, edge_weights = graphs.get_adjacency()
            
            assert row_idxs.shape[0] < 26 * 26,\
                "Expected an array of at most 26 x 26 for a fully connected graph"
            assert col_idxs.shape[0] < 26 * 26,\
                "Expected an array of at most 26 x 26 for a fully connected graph"
            assert edge_weights.shape[0] < 26 * 26,\
                "Expected an array of at most 26 x 26 for a fully connected graph"
            
            #bad_indices = [graphs.td_brain_channels.index(bad) for bad in bads]
            #assert bads_found[temp_file] == bads,"Should correctly load the forced bads"

    def create_temp_preprocessed_object(self, 
                                        temp_dir : str, 
                                        bads : list[str], 
                                        file_id: str = None ) -> None:
        """Helper function to create temporary preprocessed objects.
        
        Args:
        ----
            temp_dir : Path to a temporary directory to save the Preprocessed object.
            bads : List of bad channels to force.
            file_id (optional) : a new file id to give the modified Preprocessed object
        Returns:
        ----
            None
        """
        test_file: str = os.environ.get('TEST_FILE')
        if file_id:
            test_file = file_id
        test_cleaned_file: str = os.environ.get('EEG_CLEANED_TEST_FILE')
        participant : str = ""
        condition : str = ""
        participant, condition, session = get_details_from_file_name(test_file)
        preprocessed : Preproccesing = np.load(test_cleaned_file,                         
                            allow_pickle = True)
        temp_dir : Path = Path(temp_dir) / "cleaned"
        temp_dir.mkdir(parents=True,  exist_ok = True)

        preprocessed.bad_channels_after_interpolation['bad_all'] = bads
        file_name : str = \
            f'{participant}_{session}_task-rest{condition}_preprocessed.npy'
        save_path : Path = temp_dir / participant / session / "eeg"
        save_path.mkdir(parents=True,exist_ok = True)
        with open(save_path / file_name , 'wb') as output:   
            pickle.dump(preprocessed, output, pickle.HIGHEST_PROTOCOL)
        assert os.path.exists(save_path/file_name)
        return file_name, temp_dir
        

            
