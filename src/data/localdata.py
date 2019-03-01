"""
Custom dataset processing/generation functions should be added to this file
"""

__all__ = ['process_yelp'
]

import pandas as pd
import numpy as np
from ..paths import interim_data_path
from .utils import reservoir_sample
import random

def yelp_json_to_df(filename):
    """Convert a yelp JSON file to a dataframe
    
    Yelp data consists of one json-object per line.
    
    Parameters
    ----------
    
    Returns
    -------
    dataframe corresponding to the file
    """
    with open(filename, 'r') as f:
        data = f.readlines()

    # remove the trailing CR from each line
    data = map(lambda x: x.rstrip(), data)

    # Pack it up as a json list
    data_json_str = "[" + ','.join(data) + "]"

    # and load it into pandas
    df = pd.read_json(data_json_str)
    return df

def process_yelp(dataset_name='yelp', metadata=None, num_reviews=None, filename=None, random_seed=None):
    """Convert raw yelp data to a Dataset Options dictionary
    
    Parameters
    ----------
    num_reviews: int or None
        if set, randomly sample this many reviews from the yelp data
    random_seed: int
        Set for reproducible randomness
    filename: string
        A filename containing yelp accademic dataset review. This path must be fully qualified.
    """
    if metadata is None:
        metadata = {}
    if filename is None:
        filename = interim_data_path / dataset_name / 'yelp_academic_dataset_review.json'
    else:
        filename = pathlib.Path(filename)
    
    if random_seed is not None:
        metadata['random_seed'] = random_seed
        random.seed(random_seed)
    
    if num_reviews is None:
        df = yelp_json_to_df(filename)
    else:
        sample = reservoir_sample(filename, num_reviews, random_seed=random_seed)
        data_json_str = "[" + ','.join(sample) + "]"
        df = pd.read_json(data_json_str)
    
    data = np.array(df.text)
    
    target = None
    return data, target, metadata

        
