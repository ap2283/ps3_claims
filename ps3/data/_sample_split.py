import hashlib

import numpy as np

import pandas as pd

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df: pd.DataFrame, id_column: str, training_frac: float = 0.8) -> pd.DataFrame:
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    bins = ['train', 'test']

    # Convert training fraction to integer
    training_frac = int(training_frac*100)
    
    # Function to hash a value and convert to an integer
    def hash_to_int(value):
        hash_obj = hashlib.sha256(str(value).encode('utf-8'))
        hex_digest = hash_obj.hexdigest()
        return int(hex_digest, 16)  # convert hex string → integer

    # Create hash column in dataframe
    df['hash'] = df[id_column].apply(hash_to_int)
    
    # Function to assign split
    def assign_split(hash_value: int):
        bucket = hash_value % 100
        return 'train' if bucket < training_frac else 'test'

    df['sample'] = df[id_column].apply(assign_split)

    return df
