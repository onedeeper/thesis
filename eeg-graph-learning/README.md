# EEG Graph Learning

Graph-based machine learning for EEG data analysis using self-supervised learning approaches.

## Project Overview

This repository contains the implementation of graph-based machine learning models for self-supervised pre-training on EEG data. The goal is to develop robust representations of EEG signals that can be used for downstream tasks such as psychiatric condition classification.

## Installation

### Clone the repository
`git clone https://github.com/yourusername/eeg-graph-learning.git `  \

`cd eeg-graph-learning`

###  Create and activate the conda environment
`conda env create -f environment.yml`
`conda activate eeg-graph-learning`

Install the package in development mode \
`pip install -e .`


## Data Setup

This project uses the TD-Brain dataset, which requires a Data Usage Agreement (DUA).

1. Apply for access to the TD-Brain dataset : https://www.brainclinics.com/resources/tdbrain-dataset
2. Once approved, download the dataset
3. Place the data in the `data/` directory following the structure below:

![Alt text](data_dir_structure.png)

