import os
import re

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