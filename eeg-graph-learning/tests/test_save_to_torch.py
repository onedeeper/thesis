import os
import numpy as np
import pickle
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from matplotlib.figure import Figure
import pandas as pd
from pathlib import Path


from eeglearn.preprocess.preprocessing import Preproccesing
from eeglearn.preprocess import plotting
from eeglearn.preprocess.preprocess_pipeline import preprocess_pipeline
from eeglearn.preprocess.save_to_torch import get_filepaths

def test_get_filepaths():
    """
    Test the get_filepaths function
    """
    # get sample eeg from environment variable
    eeg_dir = os.environ.get('EEG_TEST_FILE_PATH')
    sample_ids = ["sub-19694366"]
    recording_condition = "EO"
    filepaths = get_filepaths(eeg_dir, sample_ids, recording_condition)
    assert len(filepaths) == 1
    assert filepaths[0].endswith(".npy")
    assert "sub-19694366" in filepaths[0]
    assert "EO" in filepaths[0]
    assert "ses-1" in filepaths[0]

if __name__ == '__main__':
    test_get_filepaths()