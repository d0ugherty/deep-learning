import argparse
import os
import numpy as np
import pandas as pd
import mne

#sample_data_folder = mne.datasets.sample.data_path()
#sample_data_raw_file = os.path.join(
 #   sample_data_folder, "MEG", "sample", "sample_audvis_raw.fif"
#)

def read_gdf_header(file_path):
    with open(file_path, 'rb') as f:
     # Step 2: Read and Parse Header
    # Read the version information (first 8 bytes in the header)
        version_info = f.read(8).decode('utf-8').strip()
        print(f"Version: {version_info}")
"""
    Convert file data 
"""


"""
    Save converted file data
"""
"""
    Read file into python
    Takes a directory as a command line argument and iterates over 
    each .gdf file to convert it into a python-readable format
"""

read_gdf_header("data/a/A01E.gdf")