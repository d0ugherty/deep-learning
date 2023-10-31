import argparse
import os
from scipy.io import loadmat
import pandas as pd


"""
    Read file into python
"""
parser = argparse.ArgumentParser(description='Convert MATLAB .gdf files in a directory to a Python-readable format.')
parser.add_argument('dir_path', type=str, help='The path of the directory containing .gdf or .mat files to convert.')

args = parser.parse_args()
dir_path = args.dir_path

file_list = os.listdir(dir_path)

for file in file_list:
  #  print(file)
    pass
"""
    Convert file data 
"""


"""
    Save converted file data
"""