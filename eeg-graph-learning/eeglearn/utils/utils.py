import os
import re
import pandas as pd
from pathlib import Path

def get_participant_id_condition_from_string(file_path):
    """
    Extract the participant ID and condition from the file path.
    """
    # Extract the participant ID and condition from the file path

    participant_match = re.search(r'(sub-\d+)(?=_)', file_path)
    participant_id = participant_match.group(1) if participant_match else None
    
    # Extract condition (restEC or restEO) from the task- portion
    condition_match = re.search(r'task-(restE[CO])', file_path)
    condition = condition_match.group(1) if condition_match else None

    return participant_id, condition

def get_labels_dict() -> dict[str, str]:
    """
    Get indications for each participant from the details excel file.
    """
    # find the path to the labels file data
    labels_file = Path(__file__).resolve().parent.parent.parent / 'data' / 'TDBRAIN_participants_V2.xlsx'
    labels_df = pd.read_excel(labels_file)
    participant_ids = labels_df['participants_ID']
    participant_labels = labels_df['indication']
    return dict(zip(participant_ids, participant_labels))