"""Tests for the Energy module.

This module contains test cases for the Energy class and its functionality including:
- Energy initialization and configuration
- Energy calculation from EEG data
- Frequency band permutations
- Spatial permutations
- Parallel processing capabilities
"""

import os
import pickle
import random
import tempfile
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
import torch

from eeglearn.config import Config
from eeglearn.features.energy import Energy
from eeglearn.preprocess.preprocessing import Preproccesing
from eeglearn.utils.utils import get_participant_id_condition_from_string, hamming_set
from unittest.mock import patch
import multiprocessing

Config.RANDOM_SEED = 1223333
Config.set_global_seed()

TEST_FILE : str = os.environ.get('TEST_FILE')

class TestEnergy:
    """Test class for the Energy feature extraction functionality.

    This class contains test cases that verify the functionality of the Energy class, 
    including:
    - Energy matrix initialization and configuration
    - Data shape and value validation
    - Frequency band permutation generation
    - Spatial permutation handling
    - Parallel processing capabilities
    - File saving and loading operations
    """

    @pytest.fixture(autouse = True)
    def setup(self):
        """Create necessary objects to be used later in testing.
        
        Args:
        ----
            None
            
        Returns:
        -------
            None
                
        Note:
        ----
            None

        """
        self.test_dir : str = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
        self.test_file : str = os.environ.get('TEST_FILE')
        self.electrode_names : list['str'] = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 
                                              'F8', 'FC3','FCz','FC4', 'T7', 'C3',
                                              'Cz', 'C4', 'T8', 'CP3','CPz', 'CP4',
                                              'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 
                                                'Oz', 'O2']
        
        self.regions : dict[str,list[str]] = {
            "pre_frontal" :["Fp1", "Fp2"],
            "frontal" :["Fz", "FCz"],
            "left_frontal" :  ["F7", "F3", "FC3"],
            "right_frontal" : ["F4", "F8", "FC4"],
            "left_temporal" : ["T7", "C3", "CP3"],
            "right_temporal" : ["C4" , "T8", "CP4"],
            "central" : ["Cz", "CPz", "Pz"],
            "left_parietal" : ["P7", "P3"],
            "right_parietal" : ["P4", "P8"],
            "occipital" : ["O1","Oz", "O2"]
            }
        
        assert len([ch for region,channels in 
                                    self.regions.items() for ch in channels]) == 26,\
                                    "Incorrect number of total channels"
        self.set_from_regions_dict = {ch for region,channels in 
                                    self.regions.items() for ch in channels}
        assert len({ch for region,channels in self.regions.items()\
                        for ch in channels}) ==26, "Not expected number of channels"
        assert len(set(self.electrode_names).difference\
                   (self.set_from_regions_dict))== 0,\
            "Not all defined channels from data are in the regions"
        

    @pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
    def test_get_energy_initialization(self) -> None:
        """Test Energy class initialization with test data."""
        # Initialize Energy with the test directory
        energy : Energy = Energy(cleaned_path=self.test_dir,
                    select_freq_bands=['delta', 'theta', 'beta', 'gamma'],
                    energy_plots=False,
                    include_bad_channels_psd=True)
        
        # Check that initialization correctly set attributes
        assert energy.cleaned_path == self.test_dir
        assert len(energy.select_freq_bands) ==4  # Should have 5 frequency bands
        assert energy.include_bad_channels_psd is True
        assert len(energy.participant_npy_files) > 0

        # test with bands = None
        # Initialize Energy with the test directory
        energy : Energy = Energy(cleaned_path=self.test_dir,
                    select_freq_bands=None,
                    energy_plots=False,  
                    include_bad_channels_psd=True)
        
        assert energy.cleaned_path == self.test_dir
        assert len(energy.select_freq_bands) == 5  # Should have 5 frequency bands
        assert energy.include_bad_channels_psd is True
        assert len(energy.participant_npy_files) > 0



    @pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
    def test_get_energy_len(self)-> None:
        """Test if the len method returns the number of the files to be proccessed.

        This wil eventually be the number of energy objects generated. 
        """
        # Initialize Energy with the test directory
        energy : Energy = Energy(cleaned_path=self.test_dir,
                    select_freq_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                    full_time_series=True,
                    energy_plots=False,
                    include_bad_channels_psd=False)
        assert len(energy) > 0, "No files found"


    @pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
    def test_get_energy_item(self)-> None:
        """Tests if the __getitem__method returns a processed energy object."""
        # Initialize Energy with the test directory
        energy : Energy = Energy(cleaned_path=self.test_dir,
                    select_freq_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                    full_time_series=True,
                    energy_plots=False,
                    verbose_psd=False,
                    include_bad_channels_psd=False)
        assert energy[0][0][0].shape[0] == 26
        assert energy[0][0][0].shape[1] == 5

    @pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment var iable not set")
    def test_get_energy_shape(self)-> None:
        """Test that get_energy returns the correct shape of energy matrix."""
        # Get path from environment variable
        
        # Initialize Energy with the test directory
        energy = Energy(cleaned_path=self.test_dir,
                    select_freq_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                    full_time_series=True,
                    energy_plots=False,
                    verbose_psd=False,
                    include_bad_channels_psd=True)
        
        # Skip if no files
        if len(energy.participant_npy_files) == 0:
            pytest.skip("No .npy files found in the test directory")
        
        # Get energy for the first file
        folder_path : Path
        file_name : str 
        folder_path, file_name = energy.folders_and_files[0]
        band_details : tuple[torch.Tensor, list[str]] = energy.get_energy(folder_path, 
                                                                        file_name)
        band_matrix : torch.Tensor = band_details[0]
        # Check shape: should be (n_channels, n_select_freq_bands)
        assert isinstance(band_matrix, torch.Tensor),\
            "Energy matrix should be a torch.Tensor"
        assert band_matrix.shape[1] == len(energy.select_freq_bands), \
            "Should have 5 frequency bands"
        assert band_matrix.shape[0] > 0, "Should have at least one channel"

        # Test epoched energy
        energy = Energy(cleaned_path=self.test_dir,
                    select_freq_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                    full_time_series=False,
                    energy_plots=False,
                    verbose_psd=False,
                    include_bad_channels_psd=True)
        band_details = energy.get_energy(folder_path, file_name)
        band_matrix =  band_details[0]
        assert isinstance(band_matrix, torch.Tensor), \
            "Energy matrix should be a torch.Tensor"
        assert band_matrix.shape[2] == len(energy.select_freq_bands), \
            f"Should have {len(energy.select_freq_bands)} frequency bands"
        assert band_matrix.shape[1] > 0, "Should have at least one channel"

    @pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
    def test_get_energy_values(self)-> None:
        """Test that get_energy returns valid energy values."""
        dir_path : str = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
        bands : list[str] = ['delta', 'theta', 'alpha', 'beta', 'gamma'] 

        # test everything with a few different random band configurations
        for _ in range(10):
            # test with full time series , bad channels included
            random_n_bands : list[str] = bands[:random.randint(1,4)]
            energy = Energy(cleaned_path=dir_path,
                        select_freq_bands= random_n_bands,
                        full_time_series= True,
                        energy_plots=False,
                        verbose_psd=False,
                        include_bad_channels_psd=True)
            
            band_details : tuple[torch.Tensor, list[str]] =  energy.get_energy\
                                                    (folder_path=Path(dir_path) \
                                    / "sub-19740274" / "ses-1" / "eeg" ,
                                    file_name= TEST_FILE)
            
            energy_data_ordered  = band_details[0]
            
            assert isinstance(energy_data_ordered,torch.Tensor),\
                "Should be a torch tensor"
            assert energy_data_ordered.shape[0] ==  26
            assert energy_data_ordered.shape[1] == len(random_n_bands)

            # it is imperative that the bands are always in the same order in the 
            # returned matrix, regardless of the order the user sends them at 
            # initialization
            random.shuffle(random_n_bands)
            energy_data_diff_order : Energy = Energy(cleaned_path=dir_path,
                        select_freq_bands=random_n_bands,
                        full_time_series= True,
                        energy_plots=False,
                        verbose_psd=False,
                        include_bad_channels_psd=True)
            band_details = energy_data_diff_order.\
                                        get_energy(folder_path=Path(dir_path) \
                                    / "sub-19740274" / "ses-1" / "eeg" ,
                                    file_name= TEST_FILE)
            data_diff_order = band_details[0]
            assert torch.allclose(data_diff_order, energy_data_ordered),\
            "Matrix was not built consistently when bands are shuffled"

    def test_parallel_returns(self) -> None:
        """Test that the parallel method returns the correct number of files."""
        dir_path : str = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
        energy : Energy = Energy(cleaned_path=dir_path,
                                select_freq_bands=['delta', 'theta',
                                                    'alpha', 'beta', 'gamma'],
                                full_time_series=True,
                                save_to_disk=False)
        files = energy.run_energy_parallel()
        assert len(files) == 1

    @pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                        reason=\
                        "EEG_TEST_CLEANED_FOLDER_PATH environment variable notset")
    @pytest.mark.skipif(not os.environ.get('EEG_CLEANED_TEST_FILE'), 
                        reason=\
                            "EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
    def test_get_freq_permutations_full_time_series(self)-> None:
        """Test case for generating the energy permutations for a given subject."""
        test_cleaned_file = os.environ.get('EEG_CLEANED_TEST_FILE')
        participant : str = ""
        condition : str = ""
        participant, condition = get_participant_id_condition_from_string(TEST_FILE)
        preprocessed : Preproccesing = np.load(test_cleaned_file, allow_pickle = True)
        test_bands : list[str] = ['gamma', 'delta']
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Created temporary directory at: {temp_dir}")
            temp_dir : Path = Path(temp_dir) / "cleaned"
            temp_dir.mkdir(parents=True,  exist_ok = True)
            
            # hard set a bad channel and save it
            preprocessed.preprocessed_raw.info['bads'] = ["F7"]
            file_name : str = \
                f'{participant}_ses-1_task-rest{condition}_preprocessed.npy'
            save_path : Path = temp_dir / participant / "ses-1" / "eeg"
            save_path.mkdir(parents=True,exist_ok = True)
            with open(save_path / file_name , 'wb') as output:   
                pickle.dump(preprocessed, output, pickle.HIGHEST_PROTOCOL)
            assert os.path.exists(save_path/file_name)
            # Test with bad channels excluded
            #--------------------------------#
            energy : Energy = Energy(cleaned_path=temp_dir,
                                select_freq_bands=test_bands,
                                full_time_series=True,
                                save_to_disk=False,
                                include_bad_channels_psd=False)
            
            band_position : dict = {band : i for i, band \
                                    in enumerate(energy.select_freq_bands)}
            
            possible_perms : dict[int, tuple[str, str, str,str,str]] =  \
                dict(enumerate(permutations(energy.select_freq_bands)))
            # testing the contents
            # The function should not return the same pseud-label each time.
            for _ in range(10):
                band_details : tuple[torch.Tensor, list[str]] = energy.get_energy\
                                                            (folder_path=temp_dir \
                                                        / participant / "ses-1" / "eeg",
                                                        file_name= TEST_FILE)
                input_matrix : torch.Tensor  = band_details[0]

                permutations_label : tuple[torch.Tensor,
                                    int] = energy.get_freq_permutation(input_matrix)
                permuted_data : torch.Tensor  = permutations_label[0]
                pseudo_label : int = permutations_label[1]     
                # testing the objects 
                # easiest to pass!
                assert isinstance(permutations_label,tuple)
                assert isinstance(permuted_data, torch.Tensor)
                assert isinstance(pseudo_label, int)

                if pseudo_label != 0:
                    assert not (torch.allclose(input_matrix,permuted_data)),\
                    "Not shuffled. Columns are the same."
                else:
                    assert torch.allclose(input_matrix,permuted_data),\
                    "Shuffled. Columns should be the same for this pseudo label"

            permutations_label : tuple[torch.Tensor,
                                    int] = energy.get_freq_permutation(input_matrix)
            permuted_data : torch.Tensor  = permutations_label[0]
            pseudo_label : int = permutations_label[1]
            
            # we test if the resulting matrix and pseudo label match the ones generated
            expected_permutation : list[str] = possible_perms[pseudo_label]

            band_ordering : list[int] = [band_position[band]\
                                        for band in expected_permutation]
            permuted_input_matrix : torch.Tesor = input_matrix[:,band_ordering]

            assert torch.allclose(permuted_data,permuted_input_matrix),\
            "The expected permutation has not been applied"

            # Test with bad channels included
            #--------------------------------#
            energy : Energy = Energy(cleaned_path=temp_dir,
                                select_freq_bands=test_bands,
                                full_time_series=True,
                                save_to_disk=False,
                                include_bad_channels_psd=True)

            # testing the contents
            # The function should not return the same pseud-label each time.
            for _ in range(10):
                band_details = energy.get_energy(folder_path=temp_dir \
                                / participant / "ses-1" / "eeg" ,
                                file_name= TEST_FILE)
                input_matrix = band_details[0]

                permutations_label : tuple[torch.Tensor,
                                    int] = energy.get_freq_permutation(input_matrix)
                permuted_data : torch.Tensor  = permutations_label[0]
                pseudo_label : int = permutations_label[1]     
                # testing the dimensions 
                # easiest to pass!
                assert isinstance(permutations_label,tuple)
                assert isinstance(permuted_data, torch.Tensor)
                assert isinstance(pseudo_label, int)

            
                if pseudo_label != 0:
                    assert not (torch.allclose(input_matrix,permuted_data)),\
                    "Not shuffled. Columns are the same."
                else:
                    assert torch.allclose(input_matrix,permuted_data),\
                    "Shuffled. Columns should be the same for this pseudo label"

            permutations_label : tuple[torch.Tensor,
                                    int] = energy.get_freq_permutation(input_matrix)
            permuted_data : torch.Tensor  = permutations_label[0]
            pseudo_label : int = permutations_label[1]
            
            # we test if the resulting matrix and pseudo label match the ones generated
            expected_permutation : list[str] = possible_perms[pseudo_label]

            band_ordering : list[int] = [band_position[band]\
                                        for band in expected_permutation]
            permuted_input_matrix : torch.Tesor = input_matrix[:,band_ordering]

            assert torch.allclose(permuted_data,permuted_input_matrix),\
            "The expected permutation has not been applied"

    def test_get_freq_permutations_epoched(self)-> None:
        """Test the permutation generation with epoched data."""
        test_cleaned_file = os.environ.get('EEG_CLEANED_TEST_FILE')
        participant : str = ""
        condition : str = ""
        participant, condition = get_participant_id_condition_from_string(TEST_FILE)
        preprocessed : Preproccesing = np.load(test_cleaned_file, allow_pickle = True)
        
        test_bands : list[str] = ['gamma', 'delta']
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Created temporary directory at: {temp_dir}")
            temp_dir : Path = Path(temp_dir) / "cleaned"
            temp_dir.mkdir(parents=True,  exist_ok = True)
            
            # hard set a bad channel and save it
            preprocessed.preprocessed_raw.info['bads'] = ["F7"]
            file_name : str = \
                f'{participant}_ses-1_task-rest{condition}_preprocessed.npy'
            save_path : Path = temp_dir / participant / "ses-1" / "eeg"
            save_path.mkdir(parents=True,exist_ok = True)
            with open(save_path / file_name , 'wb') as output:   
                pickle.dump(preprocessed, output, pickle.HIGHEST_PROTOCOL)
            assert os.path.exists(save_path/file_name)

            # Test with bad channels excluded
            #--------------------------------#
            energy : Energy = Energy(cleaned_path=temp_dir,
                                select_freq_bands=test_bands,
                                full_time_series=False,
                                save_to_disk=False,
                                include_bad_channels_psd=False)
            
            band_position : dict = {band : i for i, band \
                                    in enumerate(energy.select_freq_bands)}
            
            possible_perms : dict[int, tuple[str, str, str,str,str]] =  \
                dict(enumerate(permutations(energy.select_freq_bands)))
            # testing the contents
            # The function should not return the same pseud-label each time.
            band_details : tuple[torch.Tensor, list[str]] = energy.get_energy\
                                    (folder_path=temp_dir \
                                / participant / "ses-1" / "eeg" ,
                                file_name= TEST_FILE)
            input_matrix : torch.Tensor  = band_details[0]
            for _ in range(10):
                permutations_label : tuple[torch.Tensor,
                                    int] = energy.get_freq_permutation(input_matrix)
                permuted_data : torch.Tensor  = permutations_label[0]
                pseudo_label : int = permutations_label[1]     
                # testing the dimensions 
                # easiest to pass!
                assert isinstance(permutations_label,tuple)
                assert isinstance(permuted_data, torch.Tensor)
                assert isinstance(pseudo_label, int)

                if pseudo_label != 0:
                    assert not (torch.allclose(input_matrix,permuted_data)),\
                    "Not shuffled. Columns are the same."
                else:
                    assert torch.allclose(input_matrix,permuted_data),\
                    "Shuffled. Columns should be the same for this pseudo label"

            permutations_label : tuple[torch.Tensor,
                                    int] = energy.get_freq_permutation(input_matrix)
            permuted_data : torch.Tensor  = permutations_label[0]
            pseudo_label : int = permutations_label[1]
            
            # we test if the resulting matrix and pseudo label match the ones generated
            expected_permutation : list[str] = possible_perms[pseudo_label]

            band_ordering : list[int] = [band_position[band]\
                                        for band in expected_permutation]
            permuted_input_matrix : torch.Tesor = input_matrix[:,:,band_ordering]

            assert torch.allclose(permuted_data,permuted_input_matrix),\
            "The expected permutation has not been applied"

            # Test with bad channels included
            #---------------------------------#
            energy : Energy = Energy(cleaned_path=temp_dir,
                                select_freq_bands=test_bands,
                                full_time_series=False,
                                save_to_disk=False,
                                include_bad_channels_psd=True)

            # testing the contents
            # The function should not return the same pseud-label each time.
            for _ in range(10):
                band_details = energy.get_energy\
                                    (folder_path=temp_dir \
                                / participant / "ses-1" / "eeg" ,
                                file_name= TEST_FILE)
                input_matrix = band_details[0]

                permutations_label : tuple[torch.Tensor,
                                    int] = energy.get_freq_permutation(input_matrix)
                permuted_data : torch.Tensor  = permutations_label[0]
                pseudo_label : int = permutations_label[1]     
                # testing the dimensions 
                # easiest to pass!
                assert isinstance(permutations_label,tuple)
                assert isinstance(permuted_data, torch.Tensor)
                assert isinstance(pseudo_label, int)

                if pseudo_label != 0:
                    assert not (torch.allclose(input_matrix,permuted_data)),\
                    "Not shuffled. Columns are the same."
                else:
                    assert torch.allclose(input_matrix,permuted_data),\
                    "Shuffled. Columns should be the same for this pseudo label"

            permutations_label : tuple[torch.Tensor,
                                    int] = energy.get_freq_permutation(input_matrix)
            permuted_data : torch.Tensor  = permutations_label[0]
            pseudo_label : int = permutations_label[1]
            
            # we test if the resulting matrix and pseudo label match the ones generated
            expected_permutation : list[str] = possible_perms[pseudo_label]

            band_ordering : list[int] = [band_position[band]\
                                        for band in expected_permutation]
            permuted_input_matrix : torch.Tesor = input_matrix[:,:,band_ordering]

            assert torch.allclose(permuted_data,permuted_input_matrix),\
            "The expected permutation has not been applied"

    def test_save_freq_perms_to_disk(self)-> None:
        """Test if the generated permuations are saved to disk correctly if asked."""
        test_cleaned_file = os.environ.get('EEG_CLEANED_TEST_FILE')
        participant : str = ""
        condition : str = ""
        participant, condition = get_participant_id_condition_from_string(TEST_FILE)
        preprocessed : Preproccesing = np.load(test_cleaned_file, allow_pickle = True)
        test_bands : list[str] = ['gamma', 'delta']

        project_root : Path = Path(__file__).resolve().parent.parent.parent
        print("root: " ,project_root)

        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Created temporary directory at: {temp_dir}")
            temp_dir_cleaned : Path = Path(temp_dir) / "cleaned"
            temp_dir_cleaned.mkdir(parents=True,  exist_ok = True)
            perm_save_dir =  Path(temp_dir) / "perm_save"
            perm_save_dir.mkdir(parents=True, exist_ok=True)

            # hard set a bad channel and save it
            preprocessed.preprocessed_raw.info['bads'] = ["F7"]
            file_name : str = \
                f'{participant}_ses-1_task-rest{condition}_preprocessed.npy'
            save_path : Path = temp_dir_cleaned / participant / "ses-1" / "eeg"
            save_path.mkdir(parents=True,exist_ok = True)
            with open(save_path / file_name , 'wb') as output:   
                pickle.dump(preprocessed, output, pickle.HIGHEST_PROTOCOL)
            assert os.path.exists(save_path/file_name)

            energy : Energy = Energy(cleaned_path=temp_dir_cleaned,
                                select_freq_bands=test_bands,
                                full_time_series=False,
                                save_to_disk=False,
                                include_bad_channels_psd=False)
            
            # testing the contents
            # The function should not return the same pseud-label each time.
            band_details : tuple[torch.Tensor, list[str]] = energy.get_energy\
                                    (folder_path=temp_dir_cleaned \
                                / participant / "ses-1" / "eeg" ,
                                file_name= TEST_FILE)
            input_matrix : torch.Tensor  = band_details[0]
        
            permutations_label : tuple[torch.Tensor,
                                int] = energy.get_freq_permutation(input_matrix,
                                                            file_name="test.pt")
            permuted_data : torch.Tensor  = permutations_label[0]
            pseudo_label : int = permutations_label[1]     

            assert isinstance(permutations_label,tuple)
            assert isinstance(permuted_data, torch.Tensor)
            assert isinstance(pseudo_label, int)
            root_extension : str = "eeg-graph-learning"
            assert os.path.exists(project_root / root_extension / 'data' / 'energy'/
                                'epoched_perms' / "energy_perms_test.pt")
            file_name = "energy_perms_test.pt"
            reloaded_data = torch.load(project_root / root_extension / 'data'/\
                                'energy' / 'epoched_perms' / file_name)
            assert reloaded_data[0].shape == permuted_data.shape

            # Test with full time series
            energy : Energy = Energy(cleaned_path=temp_dir_cleaned,
                                select_freq_bands=test_bands,
                                full_time_series=True,
                                save_to_disk=False,
                                include_bad_channels_psd=False)
            

            # testing the contents
            # The function should not return the same pseud-label each time.
            band_details = energy.get_energy(folder_path=temp_dir_cleaned \
                                / participant / "ses-1" / "eeg" ,
                                file_name= TEST_FILE)
            input_matrix = band_details[0] 
            
            permutations_label : tuple[torch.Tensor,
                                int] = energy.get_freq_permutation(input_matrix,
                                                            file_name="test.pt")
            permuted_data : torch.Tensor  = permutations_label[0]
            pseudo_label : int = permutations_label[1]     

            assert isinstance(permutations_label,tuple)
            assert isinstance(permuted_data, torch.Tensor)
            assert isinstance(pseudo_label, int)
            root_extension : str = "eeg-graph-learning"
            assert os.path.exists(project_root / root_extension / 'data' / 'energy'/
                                'perms')
            file_name = "energy_perms_test.pt"
            reloaded_data = torch.load(project_root / root_extension / 'data'/\
                                'energy' / 'perms' / file_name)
            assert reloaded_data[0].shape == permuted_data.shape

    def test_run_freq_permutations_parallel(self)-> None:
        """Compare serial and parallel computations."""
        project_root : Path = Path(__file__).resolve().parent.parent.parent
    
        test_data_dir : Path \
            = project_root / "eeg-graph-learning" / "tests"/ "test_data"/\
            "parallel_test"
        test_data_dir.mkdir(parents=True,exist_ok=True)

        cleaned_path = \
            Path(__file__).resolve().parent.parent.parent /"eeg-graph-learning"/\
            'data' / 'cleaned'
        
        dataset = Energy(cleaned_path=cleaned_path,
                        testing= True,
                        full_time_series=False,
                            energy_plots=True,
                            verbose_psd=False,
                            picks_psd = ['eeg'],
                            include_bad_channels_psd=True,
                            save_to_disk=True,
                            select_freq_bands=['gamma', 'delta',
                                                'theta','alpha','beta']) 
        dataset.energy_save_dir_epoched = test_data_dir / 'energy'
        dataset.energy_save_dir_epoched.mkdir(parents=True, exist_ok= True)
        # setting full length directory to be empty for testing purposes
        # This is because run_get_permutations_parallel() handles both epoched
        # and full timeseries data together.
        empty_dir = test_data_dir / "empty"
        empty_dir.mkdir(parents=True, exist_ok= True)
        dataset.energy_save_dir = empty_dir
        dataset.run_energy_parallel()
        results = dataset.run_freq_permutations_parallel()

        seed = 42
        ctr = 0
        for data, _, file_name in results:
            # The last file in the test runs on the same process so it 
            # becomes the next number of the sequence.
            if ctr < len(results)-1:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                ctr += 1
            band_details : tuple[torch.Tensor, list[str]] =  torch.\
                                            load(test_data_dir / "energy" / file_name)
            energy_file = band_details[0]
            data_iter, _, _ = \
                dataset.get_freq_permutation(data = energy_file) 
            assert torch.allclose(data, data_iter)
            
    def test_get_spatial_perms(self) -> None:
        """Test the generated spatial permutations.
        
        This methods setsup energy objects with both epoched and non-epoched data
        and tests the permutations for correctness.
        
        Args:
        ----
            None
            
        Returns:
        -------
            None

        """
        project_root : Path = Path(__file__).resolve().parent.parent.parent
    
        test_data_dir : Path = project_root / "eeg-graph-learning" / "tests"/\
              "test_data"/ "parallel_test"
        test_data_dir.mkdir(parents=True,exist_ok=True)

        cleaned_path = Path(__file__).resolve().parent.parent.parent /\
            "eeg-graph-learning"/\
            'data' / 'cleaned'
        
        # Full time series
        dataset : Energy = Energy(cleaned_path=cleaned_path,
                        testing= True,
                        full_time_series=True,
                            energy_plots=True,
                            verbose_psd=False,
                            picks_psd = ['eeg'],
                            include_bad_channels_psd=True,
                            save_to_disk=True,
                            select_freq_bands=['gamma', 'delta', 'theta','alpha',
                                               'beta'])
        
        save_path : Path = Path(__file__).resolve().parent.parent.parent /\
                    "eeg-graph-learning" / "tests" / "test_data"
        
        self.helper_get_permutations(dataset,save_path)
        self.helper_get_permutations(dataset, save_path, test_file_loading=True)

        # Epoched time series
        dataset : Energy = Energy(cleaned_path=cleaned_path,
                        testing= True,
                        full_time_series=False,
                            energy_plots=True,
                            verbose_psd=False,
                            picks_psd = ['eeg'],
                            include_bad_channels_psd=True,
                            save_to_disk=True,
                            select_freq_bands=['gamma', 'delta', 'theta','alpha',
                                               'beta'])
        self.helper_get_permutations(dataset,save_path)
        self.helper_get_permutations(dataset, save_path, test_file_loading=True)

    def helper_get_permutations(self,
                              dataset : Energy,
                              save_path: Path,
                              test_file_loading : bool = False):
        """Test the generated spatial permutations.
        
        Helper method that contains the testing code for permutations.
        It handles both epoched and non-epoched data generated test_get_spatial_perms
        
        Args:
        ----
            dataset : Energy object containing the information about the dataset
                      and the necessary methods.
            save_path : path to load the saved permutations data
            
        Returns:
        -------
            None

        """
        idx_to_region : dict[int,str] = dict(enumerate(list(self.regions.keys())))
        test_with_random : bool = True
        hamming_selection : str = "max"
        n_regions : str = 10
        n_permutations : str = 128
        perm_file_name : str =\
                f"{hamming_selection}_hamming_set_{n_regions}_{n_permutations}.pt"
        if not(os.path.exists(save_path / perm_file_name)):
            permutations = hamming_set(n_regions=10, n_permutations=128,\
                                        selection='max',
                                        output_file_name= perm_file_name, 
                                        save_to_disk=False)
            torch.save(torch.Tensor(permutations), save_path / perm_file_name)
        else:
            permutations = torch.load(save_path / perm_file_name)
        
        input_matrix : torch.Tensor = dataset[0][0][0] #freq bands
        ch_names : list[str] = dataset[0][0][1] # channel order info=

        if test_with_random:
            # Random matrices with varying number of epochs and channels.
            random_n_epochs :int = random.randint(1,15)
            random_n_channels : int = random.randint(1,26)
            random_n_bands : int = random.randint(1,5)
            input_matrix : torch.Tensor = torch.Tensor(np.random.random((
                                                        random_n_epochs,
                                                        random_n_channels,
                                                        random_n_bands))).float()
            ch_names = ch_names[:random_n_channels]

        output_matrix : torch.Tensor 
        pseudo_labels : torch.Tensor

        if test_file_loading:
            energy_files : list[str] = os.listdir(dataset.energy_save_dir)
            energy_files.append(os.listdir(dataset.energy_save_dir_epoched))
            random_file = energy_files[random.randint(0,len(energy_files)-1)]
            if os.path.exists(dataset.energy_save_dir_epoched / random_file):
                input_matrix, ch_names = torch.\
                    load(dataset.energy_save_dir_epoched / random_file)
            else:
                input_matrix,ch_names = torch.\
                    load(dataset.energy_save_dir / random_file)

            output_matrix, pseudo_labels = dataset.\
                    get_spatial_permutation(file_name=random_file)
        else:
            output_matrix, pseudo_labels = dataset.get_spatial_permutation(ch_names,
                                                                        input_matrix,
                                                                        )
        assert output_matrix.shape == input_matrix.shape
        assert isinstance(pseudo_labels, np.ndarray),\
            "For epoched data, tensor of permutations is expected"
        assert pseudo_labels.shape[0] == input_matrix.shape[0], \
            "Expected a pseudolabel for each epoch"
        
        shuffled_data = torch.zeros(input_matrix.shape).double()
        for epoch, pseudo_label in enumerate(pseudo_labels):
                permuted_channels : list[int] = []
                idxs_chs_in_region : dict[str, list[int]] = {}
                target_permutation : torch.Tensor = permutations[pseudo_label,:]
                for region in target_permutation:
                    channels_in_region : list[int] = \
                        self.regions[idx_to_region[region.item()]]
                    ch_idxs : list[int] = []
                    for channel in channels_in_region:
                        try:
                            permuted_channels.append(ch_names.index(channel))
                            ch_idxs.append(ch_names.index(channel))
                        except ValueError :
                            continue
                    # save the channels in each region.
                    idxs_chs_in_region[region.item()] = ch_idxs 
                assert len(permuted_channels) == input_matrix.shape[1]
                start = 0
                # test if the shuffling has been done while preserving the regions
                for region in target_permutation:
                    # check if this region is in the right place in permuted channels
                    region_size = len(idxs_chs_in_region[region.item()])
                    assert idxs_chs_in_region[region.item()] == \
                                        permuted_channels[start:start + region_size],\
                                                    "Regions are not intact."
                    start += region_size
                shuffled_data[epoch,:,:] = input_matrix[epoch,permuted_channels,:]
                
        assert torch.allclose(shuffled_data, output_matrix),\
                    "Expected permutation has not been applied."
    
def test_run_spatial_perms_parallel(self, monkeypatch):
        """
        Test if the permutations generated in parallel are as expected
        """

        # set up the necessary resources.
        project_root : Path = Path(__file__).resolve().parent.parent.parent
    
        test_data_dir : Path \
            = project_root / "eeg-graph-learning" / "tests"/ "test_data"/\
            "parallel_test"
        test_data_dir.mkdir(parents=True,exist_ok=True)

        cleaned_path = \
            Path(__file__).resolve().parent.parent.parent /"eeg-graph-learning"/\
            'data' / 'cleaned'
        
        # full time series
        dataset = Energy(cleaned_path=cleaned_path,
                        testing= True,
                                                full_time_series=False,
    energy_plots=True,
                            verbose_psd=False,
                            picks_psd = ['eeg'],
                            include_bad_channels_psd=True,
                            save_to_disk=True,
                            select_freq_bands=['gamma', 'delta',
                                                'theta','alpha','beta']) 
        dataset.energy_save_dir_epoched = test_data_dir / 'energy'
        dataset.energy_save_dir_epoched.mkdir(parents=True, exist_ok= True)
        # setting full length directory to be empty for testing purposes
        # This is because run_get_permutations_parallel() handles both epoched
        # and full timeseries data together.
        empty_dir = test_data_dir / "empty"
        empty_dir.mkdir(parents=True, exist_ok= True)
        dataset.energy_save_dir = empty_dir

        # test_matrices : list[torch.Tensor]= [torch.Tensor(np.random.random\
        #                                                   ((random.randint(1,10),
        #                          random.randint(1,26),
        #                          random.randint(1,5))))\
        #                             for _ in range(10)]
        # test_ch_names = [self.electrode_names[:matrix.shape[1]] \
        #                  for matrix in test_matrices]
        # hamming_selection : list[str] = ["max"]*10
        # test_args = zip(test_matrices,test_ch_names,hamming_selection)
        # serial_results = []
        # for item in test_args:
        #     serial_results.append(dataset.get_spatial_permutation(*item))

        empty_dir = test_data_dir / "empty"
        empty_dir.mkdir(parents=True, exist_ok= True)
        dataset.energy_save_dir = empty_dir
        dataset.run_energy_parallel()
        with patch.object(multiprocessing, "Pool") as MockPool:
            instance = MockPool.return_value
            instance.starmap.return_value = [ "foo", "bar" ]
            instance.__enter__.return_value = instance

            results = dataset.run_spatial_permutations_parallel()

            MockPool.assert_called_once()

        # seed = 42
        # ctr = 0
        # for data, _, file_name in results:
        #     # The last file in the test runs on the same process so it 
        #     # becomes the next number of the sequence.
        #     if ctr < len(results)-1:
        #         random.seed(seed)
        #         np.random.seed(seed)
        #         torch.manual_seed(seed)
        #         ctr += 1
        #     band_details : tuple[torch.Tensor, list[str]] =  torch.\
        #                                     load(test_data_dir / "energy" / file_name)
        #     energy_file = band_details[0]
        #     ch_info = band_details[1]
        #     data_iter, _, = \
        #         dataset.get_spatial_permutation(data = energy_file,
        #                                         ch_names=ch_info) 
        #     assert torch.allclose(data, data_iter)