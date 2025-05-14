from eeglearn.preprocess.preprocessing import Preproccesing
import numpy as np
import tempfile
import os
from eeglearn.utils.utils import load_preprocessed_data
import pickle
from pathlib import Path
import torch

def test_load_preprocessed_data() -> None:
    """Tests the loading of a Preprocesing object"""
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_file = os.environ.get('EEG_TEST_FILE_PATH')
        prep_object = Preproccesing(filename=csv_file)
        with open(Path(temp_dir) / 'testfile.npy', 'wb') as output:
            pickle.dump(prep_object, output, pickle.HIGHEST_PROTOCOL)

        loaded_prep = load_preprocessed_data(Path(temp_dir), 'testfile.npy')
        # test a random attribute.
        assert prep_object.bad_channels_after_interpolation ==\
            loaded_prep.bad_channels_after_interpolation
        

# def test_adjacency()-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tests the generation of the adjacency matrix"""

