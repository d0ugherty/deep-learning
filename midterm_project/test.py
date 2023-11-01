# Required Libraries
import numpy as np
import pyedflib  # Assuming EDF (European Data Format) is used, similar to the Biosig library in MATLAB

# Define the getData function
def getData(subject_index=6):
    
    # Build path to the data file
    dir_1 = f"A01E.gdf"  # Replace with your actual path
    
    # Load the data
    try:
        f = pyedflib.EdfReader(dir_1)
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
            sigbufs[i, :] = f.readSignal(i)
        f._close()
    except:
        print("An error occurred while reading the data.")
    
    # Other operations (e.g., data cleaning, transformation, etc.)
    # ...
    
    return sigbufs, signal_labels

# Example usage
data, labels = getData()
