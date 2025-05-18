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
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from eeglearn.config import Config
from eeglearn.preprocess.preprocessing import Preproccesing
from eeglearn.features.graphs import Graphs
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from eeglearn.utils.utils import get_details_from_file_name
#Config.RANDOM_SEED = 1223333
import scipy

Config.RANDOM_SEED = 121321212
Config.set_global_seed()

class TestGraphs:
    """Test class for evalutating functionality of the Graphs module."""

    @pytest.fixture(autouse= True)
    def setup(self):
        self.cleaned_data_path = \
            os.environ.get("EEG_TEST_CLEANED_FOLDER_PATH")
        self.n_test_files : int = 5
        # this would be 128 for spatial permutations
        self.n_max_test_labels : int = 120
        self.n_epochs : int = np.random.randint(1,12)
        self.n_perms_per_epoch : int = np.random.randint(2,5)
        self.n_channels :int = np.random.randint(1,26)
        self.n_bands : int = np.random.randint(1,5)
        subj_ids : list[int] = [np.random.randint(10000000,15000000)\
                                for _ in range(self.n_test_files)]
        
        conditions : list[str] = ["EO", "EC"]
        sessions : list[str] = ["ses-1", "ses-2"]
        test_file_ids : list[str] = []
        for subj_id in subj_ids:
            condition = random.sample(conditions,1)[0]
            session = random.sample(sessions,1)[0]
            id = f"band_perms_energy_sub-{subj_id}_{condition}_{session}_epoched.pt"
            test_file_ids.append(id)
        
        self.td_brain_channels : list[str] = [  'Fp1', 'Fp2', 'F7', 'F3', 
                                                'Fz', 'F4', 'F8', 'FC3', 
                                                'FCz', 'FC4', 'T7', 'C3', 
                                                'Cz', 'C4', 'T8', 'CP3',
                                                'CPz', 'CP4', 'P7', 'P3', 
                                                'Pz', 'P4', 'P8', 'O1',
                                                'Oz', 'O2']
        self.ch_names_to_idxs : list[int] = { ch : idx 
                                             for idx, ch in 
                                                    enumerate(self.td_brain_channels)}
        self.test_file_list : list[tuple[torch.Tensor,
                                         np.ndarray,
                                         str]] =[]
        for i, file_id in enumerate(test_file_ids):
            # test the noise floor filtering.
            if i == 2:
                data = torch.full((self.n_epochs,
                                self.n_perms_per_epoch,
                                self.n_channels,
                                self.n_bands), 1e-11)
                #print(data.shape)
                noise = torch.full((self.n_channels,
                                self.n_bands), 1e-11)
                data[0,1,:,:] = noise
            else:
                data = torch.rand((self.n_epochs,
                                self.n_perms_per_epoch,
                                self.n_channels,
                                self.n_bands))
            pseudo_labels : torch.Tensor = \
                            torch.Tensor(np.random.randint(0,self.n_max_test_labels, 
                              size = (self.n_epochs, self.n_perms_per_epoch)))
            self.test_file_list.append((data,pseudo_labels,file_id))
        
    def test_graph_init(self) -> None:
        """Test creation of Graphs object.
        
        Args:
        ----

        Returns:
        ----
            None
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            energy_path : Path = Path(temp_dir) / "energy"
            energy_path.mkdir(parents=True,exist_ok = True)
            graphs = Graphs(perm_type="spatial",
                            distance="ellipsoid",
                            cleaned_data_path=self.cleaned_data_path,
                            energy_path=energy_path)
            assert(isinstance(graphs, Graphs))

            # should fail with strings that are not as intended
            with pytest.raises(AssertionError):
                dist = "sfe323r32asfe"
                graphs = Graphs(perm_type="spatial",
                            distance=dist,
                            cleaned_data_path=self.cleaned_data_path,
                            energy_path=energy_path)
            with pytest.raises(NotImplementedError):
                graphs = Graphs(perm_type="spatial",
                            distance="eucledian",
                            cleaned_data_path=self.cleaned_data_path,
                            energy_path=energy_path)
    def test_get_graphs(self) -> None:
        """Test the generation of graph objects.
        
        Args:
        ----
            None
        Returns:
        ----
            None
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            skip = 0
            participant_bads : dict[str,list[str]] = {}
            for data,pseudo_labels,file_id in self.test_file_list:
                bads: list[str] = random.sample(self.td_brain_channels,4)
                if skip == 1:
                    bads = []
                participant_bads[file_id] = bads
                _, temp_dir_cleaned= self.create_temp_preprocessed_object(
                                                    temp_dir=temp_dir,
                                                            bads=bads,
                                                            file_id=file_id)
                save_path : Path = Path(temp_dir) / "energy" / "spatial_perms"
                save_path.mkdir(parents=True,exist_ok = True)
                with open(save_path / file_id , 'wb') as output:   
                    torch.save((data, pseudo_labels, file_id), output)
                assert os.path.exists(save_path/file_id)
                skip += 1

            batch_size = len(self.test_file_list)

            graphs = Graphs(perm_type="spatial",
                        distance="ellipsoid",
                        cleaned_data_path=temp_dir_cleaned,
                        energy_path=f"{temp_dir}/energy",
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=False)
            
            files_to_load = [id for _,_,id in self.test_file_list]
            graph_loader = graphs.get_graphs(files_to_load=files_to_load)
            
            assert isinstance(graph_loader, DataLoader),\
                "Should return torch geometric dataloader"
            graph_loader= list(graph_loader)
            first_example = graph_loader[0]
            # each example should be a graph data object
            assert isinstance(first_example, Data), "Expected a torch geometric graph."
            all_examples = [ example for graph_batch in graph_loader
                                            for example in graph_batch.to_data_list()]
            print(f"all examples : {len(all_examples)}")
            n_per_participant : int = self.n_epochs * self.n_perms_per_epoch
            start = 0
            for p_idx, participant_end in enumerate(range(n_per_participant,
                                                          len(all_examples) +\
                                                            n_per_participant,
                                                          n_per_participant)):
                participant = all_examples[start:participant_end]
                print(f"participant : {p_idx}, n_examples : {len(participant)}")
                for graph in participant:
                    rows : torch.Tensor = graph.edge_index[0,:]
                    cols : torch.Tensor = graph.edge_index[1,:]
                    bads = participant_bads[self.test_file_list[p_idx][2]]
                    bad_idxs : torch.Tensor = torch.Tensor([self.ch_names_to_idxs[bad] 
                                                            for bad in bads])
                    assert graph.y.shape[0] == 1, "Each example should have 1 label."
                    assert graph.x.shape == (self.n_channels, self.n_bands),\
f"Participant : {p_idx} : Data matrix should be a 2d representation of an EEg signal."
                    assert not torch.all(torch.isin(rows, bad_idxs)),\
                        f"{p_idx}: None of the bad channels should have edges."      
                    csr = scipy.sparse.csr_matrix((graph.edge_attr, (rows, cols)),
                                            shape=(26, 26))
                    assert scipy.linalg.issymmetric(csr.todense()),\
                        f"{p_idx}: Expecting a symmetric distance matrix." 
                    assert np.allclose(csr.diagonal(), np.zeros(26)),\
                        f"Participant : {p_idx} : Expecting no self loops."
                    
                start = participant_end

    def test_get_ellipsoid_distance(self) -> None:
        """Test the calculation of ellipsoidal distances between electrodes.
        
        Args:
        ----

        Returns:
        ----
            None
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path : Path = Path(temp_dir) / "energy"
            save_path.mkdir(parents=True,exist_ok = True)
            graphs = Graphs(perm_type="spatial",
                            distance="ellipsoid",
                            cleaned_data_path=self.cleaned_data_path,
                            energy_path= save_path)
            distance = graphs.get_ellipsoid_distance()
            self.distance_helper(distance)

    def test_get_distance(self) -> None:
        """Test the calculation of distances between electrodes by chosen metric.
        
        Args:
        ----

        Returns:
        ----
            None
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            energy_path : Path = Path(temp_dir) / "energy"
            energy_path.mkdir(parents=True,exist_ok = True)
            graphs = Graphs(perm_type="spatial",
                            distance="ellipsoid",
                            cleaned_data_path=self.cleaned_data_path,
                            energy_path=energy_path)
            distance = graphs.get_distance()
            self.distance_helper(distance)

    def distance_helper(self, data : torch.Tensor) -> None:
        """Test the calculation of distances between electrodes by chosen metric.
           Checks for some properties we would expect from a distance matrix
           in an undirected graph.
        
        Args:
        ----
            data : A square tensor containing the edge information between each node.
        Returns:
        ----
            None
        """
        assert isinstance(data, torch.Tensor), "Expected an array"
        assert data.shape == (26,26), "Should contain distance between each node"
        assert torch.allclose(torch.diag(data), torch.zeros(26)),"Distance to self == 0"
        assert torch.allclose(data - data.T, torch.zeros(data.shape)),\
            "Should be symmetric"
        assert torch.min(data) >= 0, "Distances should be strictly greater than 0"

    def test_get_bads(self) -> None:
        """Test the loading of bad channels for a given participant and sesion.
        Args:
        ----

        Returns:
        ----
            None
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            energy_path : Path = Path(temp_dir) / "energy"
            energy_path.mkdir(parents=True,exist_ok = True)
            bads: list[str] = ["Fp1", "P3"]
            temp_file, temp_dir_updated = self.create_temp_preprocessed_object(
                                                    temp_dir=temp_dir,
                                                            bads=bads)
            graphs = Graphs(perm_type="spatial",distance="ellipsoid",
                            cleaned_data_path=temp_dir_updated,energy_path=energy_path)
            bads_found = graphs.get_bad_channels()
            assert bads_found[temp_file] ==bads,"Should correctly load the forced bads"
    
    def test_get_adjacency(self)-> None:
        """Test the generation of adjacency matrix accounting for bad channels
        
        Args:
        ----
            
        Returns:
        ----
            None
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            energy_path : Path = Path(temp_dir) / "energy"
            energy_path.mkdir(parents=True,exist_ok = True)
            bads: list[str] = ["Fp1", "Oz"]
            _, temp_dir_updated = self.create_temp_preprocessed_object(
                                                    temp_dir=temp_dir,
                                                            bads=bads)
            graphs = Graphs(distance="ellipsoid",
                            perm_type="spatial",
                    cleaned_data_path=temp_dir_updated,
                    energy_path=energy_path)
            row_idxs, col_idxs, edge_weights = graphs.get_adjacency()
            
            assert row_idxs.shape[0] < 26 * 26,\
                "Expected an array of at most 26 x 26 for a fully connected graph"
            assert col_idxs.shape[0] < 26 * 26,\
                "Expected an array of at most 26 x 26 for a fully connected graph"
            assert edge_weights.shape[0] < 26 * 26,\
                "Expected an array of at most 26 x 26 for a fully connected graph"
            csr = scipy.sparse.csr_matrix((edge_weights, (row_idxs, col_idxs)),
                                          shape=(26, 26))
            assert scipy.linalg.issymmetric(csr.todense()),\
                "Expecting a symmetric distance matrix."

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
        preprocessed : Preproccesing = np.load(test_cleaned_file, allow_pickle = True)
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
        

            
