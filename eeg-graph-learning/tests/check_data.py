import os
from pathlib import Path

def check_subject_files(cleaned_dir):
    # Create a dictionary to hold subject IDs and their conditions
    subject_conditions = {}


    original_files = os.listdir('/Users/udeshhabaraduwa/Library/CloudStorage/GoogleDrive-u.habaraduwakandambige@tilburguniversity.edu/My Drive/Tilburg/Masters CSAI/Semester 4/code/thesis/eeg-graph-learning/data/TDBRAIN-dataset/derivatives')
    cleaned_files = os.listdir('/Users/udeshhabaraduwa/Library/CloudStorage/GoogleDrive-u.habaraduwakandambige@tilburguniversity.edu/My Drive/Tilburg/Masters CSAI/Semester 4/code/thesis/eeg-graph-learning/data/cleaned')
    print(len(original_files))
    print(len(cleaned_files))
    print(original_files[0], cleaned_files[0])
    
    # check if the files are the same
    original_files = set(original_files)
    cleaned_files = set(cleaned_files)
    print(original_files - cleaned_files)
    print(cleaned_files - original_files)

if __name__ == "__main__":
    # Define the cleaned directory path
    cleaned_dir = Path(__file__).resolve().parent.parent.parent / 'eeg-graph-learning'/ 'data' / 'cleaned'
    print(cleaned_dir)
    check_subject_files(cleaned_dir)