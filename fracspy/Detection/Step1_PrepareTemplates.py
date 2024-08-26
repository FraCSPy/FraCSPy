r"""
Matched Filtering for Event Template Detection
=================================================

Overview
--------
This script implements a matched filtering technique to detect seismic events by comparing extracted templates from various stations against continuous seismic data. The goal is to identify strong correlations between the templates and segments of the continuous data.

Methodology
-----------
1. **Data Preparation**:
   - The script first reads the continuous waveform data and the previously extracted templates.
   - A band-pass filter is applied to the continuous data; the frequency range is 1-200 Hz.

2. **Template Matching**:
   - The continuous data is divided into segments that match the length of the templates.
   - For each segment, the script performs cross-correlation with each template to find the maximum correlation value.
   - The start time of the segment is calculated to provide precise temporal information regarding the detected events.

3. **Thresholding**:
   - If the maximum correlation value exceeds a predefined threshold, the results (start time, station name, and correlation value) are stored in a dictionary.

4. **Output**:
   - The results are printed to the console, and they are saved in a folder named **"DetectedEvents"**. Each detected event is saved as a NumPy file, and detailed results are stored in a JSON file.
   - Additionally, a separate JSON file containing the detected times is also created.
"""

import obspy
import numpy as np
import glob
import os
import json

###############################################################################
# Constants
# ---------------------------
NUMBER_OF_TEMPLATES = 20  # Number of templates to extract
TEMPLATE_DURATION = 4      # Duration of each template (in seconds)

###############################################################################
# Read Continuous Data and Event Catalog
# ---------------------------
# Read continuous waveform data from the specified directory
data_paths = glob.glob('data/*.mseed')

# Load the short event catalog from an Excel file
catalog = pd.read_excel('Catalog_JGR_OneDay.xlsx')

# Select the largest 20 magnitude events from the catalog
catalog = catalog.sort_values(by=['Magnitude Mw'], ascending=False).head(NUMBER_OF_TEMPLATES)

# Create a directory to save the templates if it doesn't exist
output_dir = f"database/"
os.makedirs(output_dir, exist_ok=True)
        
###############################################################################
# Process Each Continuous Data File
# ---------------------------
# Loop over each continuous data file (currently set to process the first file only)
for data_file in data_paths:  # Adjust to iterate over multiple files if necessary
    # Initialize a list to store station templates
    station_templates = []
    
    # Print the current data file being processed and the number of events to process
    print(f"Processing Data File: {data_file}, Number of Events to Process: {len(catalog)}")

    # Loop through the selected events in the catalog
    for index in range(len(catalog)):
        # Read the continuous data stream
        st = obspy.read(data_file)
        
        # Apply a band-pass filter to the data (1-200 Hz)
        st.filter('bandpass', freqmin=1, freqmax=200, corners=4, zerophase=False)
        
        # Extract the station name from the stream
        station_name = st[0].stats.station

        # Create the extraction time based on the event details in the catalog
        event_time_str = f"2016-{int(catalog.iloc[index]['Month']):02d}-{int(catalog.iloc[index]['Day']):02d}T" \
                         f"{int(catalog.iloc[index]['Hour']):02d}:{int(catalog.iloc[index]['Minute']):02d}:" \
                         f"{int(catalog.iloc[index]['Second']):02d}.000000Z"
        event_time = obspy.UTCDateTime(event_time_str)

        # Formulate the file name for storing the templates
        file_name = f"EV{int(catalog.iloc[index]['Month']):02d}{int(catalog.iloc[index]['Day']):02d}_" \
                    f"{int(catalog.iloc[index]['Hour']):02d}{int(catalog.iloc[index]['Minute']):02d}" \
                    f"{int(catalog.iloc[index]['Second']):02d}"


        # Trim the stream to the specified time window for the template
        st_trimmed = st.trim(event_time, event_time + TEMPLATE_DURATION)

        # Extract data from the first channel of the stream
        data_array = st_trimmed[0].data
        
        # Normalize the data to have a maximum absolute value of 1
        data_array /= np.max(np.abs(data_array))
        
        # Append the station data to the list of templates
        station_templates.append(data_array)

    # Save the array of station templates to a .npy file
    np.save(os.path.join(output_dir, station_name), station_templates)

