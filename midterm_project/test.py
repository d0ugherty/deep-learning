import mne
from scipy.io import savemat

# Read the .gdf file
raw = mne.io.read_raw_edf('A01E.edf',preload=False, stim_channel=None)

# Extract the data and times
data, times = raw[:, :]

# Create a dictionary to store the data and times
data_dict = {'data': data, 'times': times}

# Save as a .mat file
savemat('your_file.mat', data_dict)
