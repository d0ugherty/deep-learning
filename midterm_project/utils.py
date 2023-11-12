import mne
import os
import glob
import pandas as pd
import numpy as np
import scipy.signal as signal


def gdf_to_df(file_path):
    raw_gdf = mne.io.read_raw_gdf(file_path)  # Use the full path here
    df = raw_gdf.to_data_frame()
    return df



def gdf_to_csv(file_path):
    raw_gdf = mne.io.read_raw_gdf(file_path)  # Use the full path here
    df = raw_gdf.to_data_frame()
    file_name = file_path.replace('.gdf', '.csv')
    df.to_csv(f'csv/{file_name}', sep=',', encoding='utf-8')

def csv_to_df(file_path):
    giga_df = pd.DataFrame()

    csv_files = glob.glob(file_path)
    print("Creating dataframe...")
    for file in csv_files:
        df = pd.read_csv(file, index_col=False, dtype=float)
        giga_df = pd.concat([giga_df, df], ignore_index=True)
    
    return giga_df

def format_df(df):
    print("Formatting dataframe...")
    df = df.drop(df.columns[0], axis=1)
    df = pd.DataFrame(df.values)
    return df

def hp_filter(dataframe,sampling_rate, cutoff):
    length = len(dataframe)
    
    array = dataframe.values
    # nyquist frequency = sampling / 2
    nyq = 0.5 * sampling_rate
    # butterworth filter requires normalized values
    normalized_cutoff = cutoff / nyq
    b, a = signal.butter(1, normalized_cutoff, btype='high', analog=False)

    for i in range(length):
        array[i] = signal.filtfilt(b, a, array[i], axis=0)
