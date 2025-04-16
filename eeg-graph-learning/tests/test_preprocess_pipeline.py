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

# Import the function to test
from eeglearn.preprocess.clean import process_file, clean_pipeline


# TODO: Properly mock and test the preprocessing pipeline 
