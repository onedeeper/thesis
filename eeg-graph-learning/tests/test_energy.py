import pytest
import numpy as np
import os
from pathlib import Path
from eeglearn.features.energy import Energy
from eeglearn.utils.augmentations import create_base_raw, inject_nans
import pickle
import torch
import tempfile
import os
from eeglearn.utils.utils import get_participant_id_condition_from_string
from eeglearn.preprocess.preprocessing import Preproccesing
import random
from itertools import permutations
TEST_FILE : str = "sub-19740274_ses-1_task-restEC_preprocessed.npy"

@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_get_energy_initialization() -> None:
    """Test Energy class initialization with test data"""
    # Get path from environment variable
    test_dir : str = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize Energy with the test directory
    energy : Energy = Energy(cleaned_path=test_dir,
                   select_freq_bands=['delta', 'theta', 'beta', 'gamma'],
                   energy_plots=False,
                   include_bad_channels_psd=True)
    
    # Check that initialization correctly set attributes
    assert energy.cleaned_path == test_dir
    assert len(energy.select_freq_bands) ==4  # Should have 5 frequency bands
    assert energy.include_bad_channels_psd is True
    assert len(energy.participant_npy_files) > 0

    # test with bands = None
    # Initialize Energy with the test directory
    energy : Energy = Energy(cleaned_path=test_dir,
                   select_freq_bands=None,
                   energy_plots=False,  
                   include_bad_channels_psd=True)
    
    assert energy.cleaned_path == test_dir
    assert len(energy.select_freq_bands) == 5  # Should have 5 frequency bands
    assert energy.include_bad_channels_psd is True
    assert len(energy.participant_npy_files) > 0



@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_get_energy_len()-> None:
    """
    Test if the len method returns the number of the files to be proccessed. 
    This wil eventually be the number of energy objects generated. 
    """
    test_dir = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize Energy with the test directory
    energy : Energy = Energy(cleaned_path=test_dir,
                   select_freq_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                   full_time_series=True,
                   energy_plots=False,
                   include_bad_channels_psd=False)
    assert len(energy) > 0, "No files found"


@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_get_energy_item()-> None:
    "Tests if the __getitem__method returns a processed energy object."
    test_dir : str = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize Energy with the test directory
    energy : Energy = Energy(cleaned_path=test_dir,
                   select_freq_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                   full_time_series=True,
                   energy_plots=False,
                   verbose_psd=False,
                   include_bad_channels_psd=False)
    assert energy[0][0].shape[0] == 26
    assert energy[0][0].shape[1] == 5

@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_get_energy_shape()-> None:
    """Test that get_energy returns the correct shape of energy matrix"""
    
    # Get path from environment variable
    test_dir : str = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    
    # Initialize Energy with the test directory
    energy = Energy(cleaned_path=test_dir,
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
    band_matrix : torch.Tensor = energy.get_energy(folder_path, file_name)
    
    # Check shape: should be (n_channels, n_select_freq_bands)
    assert isinstance(band_matrix, torch.Tensor),\
        "Energy matrix should be a torch.Tensor"
    assert band_matrix.shape[1] == len(energy.select_freq_bands), \
        "Should have 5 frequency bands"
    assert band_matrix.shape[0] > 0, "Should have at least one channel"

    # Test epoched energy
    energy = Energy(cleaned_path=test_dir,
                   select_freq_bands=['delta', 'theta', 'alpha', 'beta', 'gamma'],
                   full_time_series=False,
                   energy_plots=False,
                   verbose_psd=False,
                   include_bad_channels_psd=True)
    band_matrix = energy.get_energy(folder_path, file_name)

    assert isinstance(band_matrix, torch.Tensor), \
        "Energy matrix should be a torch.Tensor"
    assert band_matrix.shape[1] == len(energy.select_freq_bands) * 12, \
        f"Should have {len(energy.select_freq_bands) * 12} frequency bands"
    assert band_matrix.shape[0] > 0, "Should have at least one channel"

@pytest.mark.skipif(not os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_get_energy_values()-> None:
    """Test that get_energy returns valid energy values.
    """
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
        
        energy_data_ordered  = energy.get_energy(folder_path=Path(dir_path) \
                                / "sub-19740274" / "ses-1" / "eeg" ,
                                file_name= TEST_FILE)
        
        assert isinstance(energy_data_ordered,torch.Tensor), "Should be a torch tensor"
        assert energy_data_ordered.shape[0] ==  26
        assert energy_data_ordered.shape[1] == len(random_n_bands)

        # it is imperative that the bands are always in the same order in the 
        # returned matrix, regardless of the order the user sends them at initialization
        random.shuffle(random_n_bands)
        energy_data_diff_order : Energy = Energy(cleaned_path=dir_path,
                    select_freq_bands=random_n_bands,
                    full_time_series= True,
                    energy_plots=False,
                    verbose_psd=False,
                    include_bad_channels_psd=True)
        data_diff_order : torch.Tensor = energy_data_diff_order.\
                                    get_energy(folder_path=Path(dir_path) \
                                / "sub-19740274" / "ses-1" / "eeg" ,
                                file_name= TEST_FILE)
        for col in range(data_diff_order.shape[1]):
                assert torch.allclose(data_diff_order[:,col], 
                                      energy_data_ordered[:,col]),\
                    "Matrix was not built consistently when bands are shuffled"

def test_parallel_returns() -> None:
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
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
@pytest.mark.skipif(not os.environ.get('EEG_CLEANED_TEST_FILE'), 
                    reason="EEG_TEST_CLEANED_FOLDER_PATH environment variable not set")
def test_get_permutations_full_time_series():
    """
    Test case for generating the energy permutations for a given subject

    What should get_permutations do ? 

    Input :  It should take a n_channels x n_bands matrix 
             1) Can have bad channels excluded so n_channels < 26
    output : It should return a permuted version of the matrix (with the rows shuffled)
             and a pseudo label for that permutation
    """
    clean_dir_path = os.environ.get('EEG_TEST_CLEANED_FOLDER_PATH')
    test_cleaned_file = os.environ.get('EEG_CLEANED_TEST_FILE')
    participant : str = ""
    condition : str = ""
    participant, condition = get_participant_id_condition_from_string(TEST_FILE)
    band_position : dict = {
        "delta" : 0,
        "theta" : 1,
        "alpha" : 2,
        "beta" :3,
        "gamma": 4,
    }
    preprocessed : Preproccesing = np.load(test_cleaned_file,                         
                           allow_pickle = True)
    bands : list[str] = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory at: {temp_dir}")
        temp_dir : Path = Path(temp_dir) / "cleaned"
        temp_dir.mkdir(parents=True, exist_ok = True)
        
        # hard set a bad channel and save it
        preprocessed.preprocessed_raw.info['bads'] = ["F7"]
        file_name : str = f'{participant}_ses-1_task-rest{condition}_preprocessed.npy'
        save_path : Path = temp_dir / participant / "ses-1" / "eeg"
        save_path.mkdir(parents=True,exist_ok = True)
        with open(save_path / file_name , 'wb') as output:   
            pickle.dump(preprocessed, output, pickle.HIGHEST_PROTOCOL)
        assert os.path.exists(save_path/file_name)
        # Energy object loading the preprocessed file with a bad channel.
        energy : Energy = Energy(cleaned_path=temp_dir,
                            select_freq_bands=['delta', 'theta',
                                                'alpha', 'beta', 'gamma'],
                            full_time_series=True,
                            save_to_disk=False,
                            include_bad_channels_psd=False)

        # testing the contents
        # The function should not return the same pseud-label each time.
        for _ in range(10):
            input_matrix : torch.Tensor  = energy.get_energy(folder_path=temp_dir \
                              / participant / "ses-1" / "eeg" ,
                               file_name= TEST_FILE)

            permutations_label : tuple[torch.Tensor,
                                int] = energy.get_permutations(input_matrix)
            permuted_data : torch.Tensor  = permutations_label[0]
            pseudo_label : int = permutations_label[1]     
            # testing the dimensions 
            # easiest to pass!
            assert isinstance(permutations_label,tuple)
            assert isinstance(permuted_data, torch.Tensor)
            assert isinstance(pseudo_label, int)

            for col in range(permutations_label[0].shape[1]):
                if pseudo_label != 0:
                    assert not (torch.allclose(input_matrix,permuted_data)),\
                    "Not shuffled. Columns are the same."
                else:
                    assert torch.allclose(input_matrix,permuted_data),\
                    "Shuffled. Columns should be the same for this pseudo label"

        permutations_label : tuple[torch.Tensor,
                                int] = energy.get_permutations(input_matrix)
        permuted_data : torch.Tensor  = permutations_label[0]
        pseudo_label : int = permutations_label[1]
        # Each permutation should have one and only one label.
        # We test if the returned permutation and its pseudolabel matches.
        # This is 1 of 120 possible permutations
        possible_perms : dict[int, tuple[str, str, str,str,str]] =  \
            {pseudo_label : perm for pseudo_label, perm \
             in enumerate(permutations(bands))}
        
        # we test if the resulting matrix and pseudo label match the ones generated
        expected_permutation : list[str] = possible_perms[pseudo_label]

        band_ordering : list[int] = [band_position[band]\
                                     for band in expected_permutation]
        permuted_input_matrix : torch.Tesor = input_matrix[:,band_ordering]

        assert torch.allclose(permuted_data,permuted_input_matrix),\
        "The expected permutation has not been applied"


# def test_get_permutations_epoched():
#     pass
