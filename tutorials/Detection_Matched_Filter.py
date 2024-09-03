r"""
Matched Filtering for Event Template Detection (Cross-correlation)
===============================================

Overview
--------
This script implements a matched filtering technique to detect seismic events by comparing extracted templates from various stations against continuous seismic data. The goal is to identify strong correlations between the templates and segments of the continuous data.

Methodology
-----------
1. **Data Preparation**:
    - The script reads continuous waveform data and the templates.
    - A band-pass filter is applied to the continuous data, with a frequency range of 1-200 Hz.

2. **Template Matching**:
    - The continuous data is divided into segments that match the length of the templates.
    - For each segment, the script performs cross-correlation with each template to find the maximum correlation value.
    - The start time of the segment is calculated to provide precise temporal information regarding the detected events.

3. **Thresholding**:
    - If the maximum correlation value exceeds a predefined threshold, the results (start time, station name, and correlation value) are stored in a dictionary.

4. **Output**:
    - Each template is saved as a NumPy file in a specified output directory.
    - A JSON file containing matched filter results for each station is saved.

Key Parameters
--------------
- **TEMPLATE_DURATION**: Duration of each template in seconds (default: 4 seconds).
- **SAMPLE_RATE**: The sampling rate of the continuous data (default: 500 Hz).
- **CORRELATION_THRESHOLD**: Threshold value for maximum correlation to filter significant results (default: 0.3).

Example: Matched Filtering for One Station
-------------------------------------------
"""

import obspy
import numpy as np
import glob
import os
import json
from fracspy.detection.matched_filtering import *

###############################################################################
# Output Directory                               
# ---------------------------                                                    
# Create an output directory for results if it doesn't exist                  
output_dir = '../data/TOC2ME_Data/output'
os.makedirs(output_dir, exist_ok=True)

###############################################################################
# Constants                                      
# ---------------------------                                                    
# Define key constants for the processing of templates and filtering            

TEMPLATE_DURATION = 4               #: Duration of each template in seconds
SAMPLE_RATE = 500                   #: Sample rate of the continuous data (assuming 500 Hz)
TEMPLATE_LENGTH = TEMPLATE_DURATION * SAMPLE_RATE + 1  #: Total number of samples per template
CORRELATION_THRESHOLD = 0.5         #: Define a threshold for maximum correlation to filter significant results
WINDOW_STEP = 1                     #: Step size for the moving window in seconds

###############################################################################
# Read Continuous Data and Template Files                   
# ---------------------------                                                    
# Read continuous data paths (mseed files) and template paths (npy files)     

data_paths = glob.glob('../data/TOC2ME_Data/*.mseed')  #: Load continuous data files
template_paths = glob.glob('../data/TOC2ME_Data/database/*.npy')  #: Load template files

###############################################################################
# Data Validation                                 
# ---------------------------                                                    
# Check to ensure that continuous data files are available for processing      
# Ensure that we have continuous data files available
if not data_paths:
    raise FileNotFoundError("No continuous data files found in the specified path.")

###############################################################################
# Process Each Continuous Data File                     
# ---------------------------                                                    
# Iterate through each continuous data file to perform matched filtering       

for first_data_file in data_paths:  # Iterate over each data file
    print(f"Testing with Continuous Data File: {first_data_file}")

    # Read the continuous data stream
    st = obspy.read(first_data_file)

    # Apply a band-pass filter to the continuous data (1-200 Hz)
    st.filter('bandpass', freqmin=1, freqmax=200, corners=4, zerophase=False)

    # Get the station name from the first trace
    station_name = st[0].stats.station
    print(f"Using Station: {station_name}")

    # Filter template paths to get only those that match the current station's templates
    specific_templates = [tp for tp in template_paths if station_name in tp]

    # Ensure we have corresponding templates for the current station
    if not specific_templates:
        print(f"No template files found for station: {station_name}")
        continue  # Skip to the next data file if no templates are available

    # Call the function to perform matched filtering
    results_dict = matched_filtering(st, specific_templates,SAMPLE_RATE,TEMPLATE_DURATION,TEMPLATE_LENGTH,WINDOW_STEP,CORRELATION_THRESHOLD)

    # Optionally save the results dictionary to a file for further analysis
    with open(os.path.join(output_dir, f'matched_filter_results_{station_name}.json'), 'w') as f:
        json.dump(results_dict, f, indent=4)

    # Print the number of detected events for the current station
    print(f"\nNumber of detected events for {station_name}: {len(results_dict)}")
