#!/usr/bin/env python3
import argparse
import os

def read_gdf_header(file_path):
    with open(file_path, 'rb') as f:
        version_info = f.read(8).decode('ascii').strip()
        print(f"Version: {version_info}")
        patient_id = f.read(80).decode('ascii').strip()
        print(f"Patient ID: {patient_id}")
        recording_id = f.read(80).decode('ISO-8859-1', errors='replace').strip()
        print(f"Recording ID: {recording_id}")
        start_date = f.read(16).decode('ascii', errors= "ignore").strip()
        print(f"Start Date: {start_date}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Convert MATLAB .gdf files in a directory to a Python-readable format.')
    parser.add_argument('dir_path', type=str, help='The path of the directory containing .gdf or .mat files to convert.')

    args = parser.parse_args()
    dir_path = args.dir_path
    
    #file_list = os.listdir(dir_path)

    # Loop Through Files
    #for file in file_list:
     #   if file.endswith('.gdf'):
      #      full_path = os.path.join(dir_path, file)
    read_gdf_header(dir_path)  
            
