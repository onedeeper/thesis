"""A class to generate graph representations of EEG energy band feature data.

Created on: May 2025
Author: Udesh Habaraduwa

Attributes
----------
dist_type : str
    The distance metric to use ("eucledian", "great_circle", or "ellipsoid").
drop_last : bool
    Whether to drop the last batch if it's smaller than `batch_size`.
shuffle : bool
    Whether to shuffle examples within each batch.
perm_type : str
    The type of permutation to be generated ("spatial" or "frequency").
batch_size : int
    Size of mini-batches for the DataLoader.
n_neighbors : int
    The number of nearest neighbors to consider for constructing graph adjacency.
n_workers : int
    Number of worker processes for data loading.
ch_positions : dict
    A dictionary mapping EEG channel names (str) to their 3D coordinates (np.array).
td_brain_channels : list[str]
    A list of standard 10-20 brain channel names used.
ch_names_to_idxs : dict
    A dictionary mapping EEG channel names (str) to their corresponding integer indices.
cleaned_data_path : str
    Path to the directory containing pre-processed EEG objects (not features).
energy_path : str
    Path to the directory containing energy features, organized by permutation type.
distances : np.ndarray
    A pre-computed matrix storing the pairwise distances between EEG channels.
base_adjacency : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    A tuple containing tensors for row indices, column indices, and edge weights
    representing the base graph adjacency.

Methods
-------
__init__(perm_type, distance, cleaned_data_path, energy_path, batch_size, n_neighbors, 
    shuffle, drop_last, n_workers)
    Initializes the Graphs object and pre-computes distances and base adjacency.
get_graphs(files_to_load)
    Generates a PyTorch Geometric DataLoader for a list of specified data files.
get_bad_channels()
    Retrieves a dictionary of bad channels for each preprocessed file.
get_distance()
    Calculates the distance matrix between EEG channels based on `dist_type`.
get_ellipsoid_distance()
    Calculates the geodesic distance between EEG channels, modeling the head as an 
    ellipsoid.
get_adjacency()
    Computes the base adjacency matrix (row, column, edge_weight) based on 
    `n_neighbors` and the derived distance.
"""
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
    def __init__(self,
                 perm_type : str ,
                 distance : str,
                 cleaned_data_path : str,
                 energy_path : str ,  
                 batch_size : int = 256,
                 n_neighbors : int = 3,
                 shuffle : bool = True,
                 drop_last : bool = True,
                 n_workers : bool = min(4,cpu_count() - 1)) -> None:
        """Generate the graph representation for a the data from participants.

        Args:
        ----
            type : spatial | frequency 
                   The type of permutation to be generated. 
            distance : "eucledian" | "great_circle" | "ellipsoid"
                       The distance metric to use when calculating the nearest 
                       neighbors for adjacency.
            cleaned_data_path : Path to pre-processed objects (not features)
            energy_path : Path to the energy folder containing the permutations
                          folders : frequency_perms or spatial_perms
            batch_size : Size of mini batches to divide the dataset into.
            n_neighbors : The number of neighbors to consider to add as adjacent
            shuffle : If the examples within a batch should be shuffled.
            drop_last : If the lat batch generated from the data should be dropped
                        if it is less than the requested batch size.
            n_workers : number of workers to use prepare batches during loading.

        Returns:
        ----
            None.
        """
        dist_types = ["eucledian", "great_circle", "ellipsoid"]
        assert distance in dist_types,\
            "Must be one of eucledian, great_circle, or ellipsoid."
        self.dist_type  : str = distance
        perm_types = ["spatial", "frequency"]
        assert perm_type in perm_types, "Must be one of spatial or frequency."
        self.drop_last : bool = drop_last
        self.shuffle : bool = shuffle
        self.perm_type = perm_type
        self.batch_size : int = batch_size
        self.n_neighbors : int = n_neighbors
        self.n_workers : int = n_workers
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
        self.td_brain_channels : list[str] = [ 'Fp1', 'Fp2', 'F7', 'F3', 
                                                'Fz', 'F4', 'F8', 'FC3', 
                                                'FCz', 'FC4', 'T7', 'C3', 
                                                'Cz', 'C4', 'T8', 'CP3',
                                                'CPz', 'CP4', 'P7', 'P3', 
                                                'Pz', 'P4', 'P8', 'O1',
                                                'Oz', 'O2']
        n_channels : int = len(self.td_brain_channels)
        self.ch_names_to_idxs : list[int] = { ch : idx 
                                             for idx, ch in 
                                                    enumerate(self.td_brain_channels)}
        assert os.path.exists(cleaned_data_path), "Cleaned directory does not exist."
        self.cleaned_data_path : str = cleaned_data_path
        assert os.path.exists(energy_path), "Energy directory does not exist."
        self.energy_path : str = energy_path
        self.distances : np.array = self.get_distance()
        assert self.distances.shape == (n_channels, n_channels),\
            "Distance matrix not as expected. Should be num_channels x num_channels."
        self.base_adjacency = self.get_adjacency()

    def get_graphs(self, files_to_load : list[str]):
        """Create the graph representations of each epoched recording in a collection.

        Args:
        ----
            files_to_load : The selected files for which to generate graphs.
                            Each should be a tuple. For example :
                            (torch.Size([12, 4, 26, 5]), <-- Data
                            (12, 4), <- pseudo_labels per epoch 
                            'energy_sub-88019481_EO_epoched.pt') <-filename
        Returns:
        ----
            Dataloader : Each mini-batch contains a 'self.batch_size' set of graphs.

        Note: This work follows https://ieeexplore.ieee.org/abstract/document/9765326
              relevant code can be found here : https://github.com/CHEN-XDU/GMSS
        """

        assert isinstance(files_to_load, list), "Expecting a list of strings."
        perm_folder : str = "frequency_perms"
        if self.perm_type == "spatial":
            perm_folder = "spatial_perms"

        files_with_bad_chs : dict[tuple[str,str,str]
                                  ,list] = {}
        for file, bads in self.get_bad_channels().items():
            if len(bads) != 0:
                participant_details = get_details_from_file_name(file)
                files_with_bad_chs[participant_details] = bads 
        
        n_epochs : int  
        n_perms_per_epoch : int 
        n_channels : int
        n_bands : int
        row : np.ndarray 
        col : np.ndarray
        edge_weight : np.ndarray
        graphs : list[Data] = []
        row, col, edge_weight = self.base_adjacency
        eps : int  = 1e-10
        noise_floor_db_scalar : float = 10 * np.log10(eps)
        for file in files_to_load:
            participant_details = get_details_from_file_name(file)
            #print(participant_details)
            permutation_data = torch.load(Path(self.energy_path) / perm_folder / file)
            examples : torch.Tensor = permutation_data[0]
            pseudo_labels : torch.Tensor = torch.Tensor(permutation_data[1])

            #print(examples.shape, pseudo_labels.shape)

            n_epochs, n_perms_per_epoch, n_channels, n_bands = examples.shape
            assert pseudo_labels.shape == (n_epochs, n_perms_per_epoch),\
                "Expected as many pseudo labels as epochs and permutations."
            
            examples = examples.reshape(n_epochs * n_perms_per_epoch, n_channels, 
                                         n_bands)
            examples = torch.unbind(examples, dim =0)
            pseudo_labels = pseudo_labels.reshape(n_epochs * n_perms_per_epoch)
            pseudo_labels = torch.unbind(pseudo_labels,dim = 0)
            bads = files_with_bad_chs.get(participant_details,None)
            #print(bads)
            if bads:
                #print()
                #bads_idx
                bad_idxs = torch.Tensor([self.ch_names_to_idxs[bad] for bad in bads])\
                    .long()
                #print(bad_idxs)
                where_bads_in_row : torch.Tensor = torch.isin(row,
                                                            torch.Tensor(bad_idxs))
                where_bad_row_idxs : torch.Tensor = torch.nonzero(where_bads_in_row).\
                                                    squeeze()
                where_bads_in_col : torch.Tensor = torch.isin(col,
                                                              torch.Tensor(bad_idxs))
                
                where_bad_col_idxs : torch.Tensor = torch.nonzero(where_bads_in_col).\
                                                      squeeze()

                mask = torch.zeros(row.shape[0], dtype=torch.bool)
                mask[where_bad_row_idxs] = True
                mask[where_bad_col_idxs] = True
                row = row[~mask].long()
                col = col[~mask].long()
                edge_weight = edge_weight[~mask]

            for i, example in enumerate(examples) :
                example_db = 10 * torch.log10(torch.clamp(example, 
                                                          min=1e-10))
                if torch.any(example_db <= noise_floor_db_scalar):
                    continue 
                edge_index = torch.vstack((row,col))
                graphs.append(Data(x= example,
                                   edge_index =  edge_index,
                                   edge_attr = edge_weight,
                                   y = pseudo_labels[i]))
                
        return DataLoader(dataset = graphs,
                          batch_size = self.batch_size,
                          shuffle = self.shuffle,
                          num_workers = self.n_workers,
                          drop_last = self.drop_last)

    def get_bad_channels(self) -> None :
        """Retreive the bad channels from preprocessed data.
        
         Args:
        ----
            None.
        Returns:
        ----
            bad_channels : dict[str,list[str]] : A dictionary keyed by file_id, with
                                            list of strings indicating which channels
                                            are bad in the file.
        """
        participant_list : list[str] = os.listdir(self.cleaned_data_path)
        folders_and_files : list[str, str] 
        folders_and_files, _ = \
            get_cleaned_data_paths(participant_list=participant_list,
                                   cleaned_path=self.cleaned_data_path)
        assert len(folders_and_files) > 0, "Atleast one cleaned file should exist."
        
        bad_channels : dict[str, list [str]] = {}
        for file in folders_and_files:
            folder_path : str = file[0]
            preprocessed_file_name : str = file[1]
            prep_data = load_preprocessed_data(folder_path=folder_path,
                                   file_name=preprocessed_file_name)
            assert isinstance(prep_data, Preproccesing),"Should be a Preprocessing obj"
            bads= prep_data.bad_channels_after_interpolation['bad_all']
            # some bad channels are returned as np.str_ class
            bad_channels[preprocessed_file_name] = [str(item) for item in bads]
        return bad_channels

    def get_distance(self) -> np.ndarray:
        """Calculates the node distance.
        
        Calculates the distance between each pair of nodes on the scalp using 
        one of three distance metrics. 

        Eucledian and great circle distance are yet to be implemented
         Args:
        ----
            None.
        Returns:
        ----
            distances : torch.Tensor : A 26x26 matrix indicating the distance between each
                                    node.
          """
        if self.dist_type == "ellipsoid":
            return self.get_ellipsoid_distance()
        else:
            raise NotImplementedError("Eucledian and great circle to be impelemtended")
    
    def get_ellipsoid_distance(self) -> np.ndarray:
        """Calculates the ellipsoidal distance between electrodes.
        
        Model the head as an ellipsoid and calculate the distance between electrodes
        as ellipsoid geodesic length. To avoid the problem discussed in [1]  using the
        Vincenty algorithm, the Karney algorithm is used.

        reference : [1] (https://ieeexplore.ieee.org/abstract/document/7833851)

         Args:
        ----
            None.
        Returns: dists : torch.Tensor : A 26x26 Tensor representing the distance
                                        between the nodes.
        ----
            None.
        """
        e = Ellipsoid()
        # Fit an ellipsoid to the position of the EEG electrodes
        fit_geo = e.best_fit(points= [values.tolist() for values in 
                                      self.ch_positions.values()])
        # center : center of the fitted ellipsoid
        # axes : The radii lengths of the 3 semi-axes
        # rot_seq : How the fitted ellipsoid is rotated in space
        (_, axes, _) = (
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
        dists = torch.Tensor(dists).float()
        assert dists.shape == (26,26), "Expect each node's distance to another."
        assert torch.all(dists >= 0), "Distances cannot be negative."
        return dists
    
    def get_adjacency(self) -> None:
        """Calculate the adajcency matrix for each participant.

        Neighbors are defined by nearest neighbors clustering. Defaults to k =3
        clustering.

        Only returns edges, not all possible positions. 

         Args:
        ----
            None.
        Returns:
        ----
            row, col : torch.Tensor : The index position in the 26x26 matrix
                                     for an edge.
            edge_wegith : torch.Tensor : The weight of the edge in that position. 
        """
        row : list[int] = []
        col : list[int] = []
        for ch_i in range(len(self.td_brain_channels)):
            neighbors = torch.argsort(self.distances[ch_i])[1:self.n_neighbors+1]
            for ch_j in neighbors:
                row.append(ch_i), col.append(ch_j.item())
        row_copy = row.copy()
        row.extend(col)
        col.extend(row_copy)
        edge_weight = np.ones(len(row), dtype=np.float32)

        assert len(edge_weight) == len(col) == len(row),\
            "Expected each edge to have a weight."
        return (torch.Tensor(row).float(),
                torch.Tensor(col).float(), 
                torch.Tensor(edge_weight).float())
    
if __name__ == "__main__":
    Config.set_global_seed()
    
    cleaned_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'cleaned'
    energy_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'energy' 
    graphs =  Graphs(
                    energy_path=energy_path,
                    distance="ellipsoid", 
                     cleaned_data_path=cleaned_path,
                     perm_type="spatial")
    graphs.get_graphs(os.listdir(energy_path / "spatial_perms"))