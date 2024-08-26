#!/usr/bin/env python
# coding: utf-8

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
   - The results are printed to the console, and they are saved in a folder named **"DetectedEvents."** Each detected event is saved as a NumPy file, and detailed results are stored in a JSON file.
   - Additionally, a separate JSON file containing the detected times is also created.
"""

###############################################################################
# Imports
# ---------------------------
import obspy
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt
from Xcorr import *

###############################################################################
# Create Output Directories
# ---------------------------
# Create an output directory for results if it doesn't exist
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

detected_events_dir = 'DetectedEvents'
os.makedirs(detected_events_dir, exist_ok=True)

###############################################################################
# Constants
# ---------------------------
# Constants
TEMPLATE_DURATION = 4               # Duration of each template in seconds
SAMPLE_RATE = 500                   # Sample rate of the continuous data (assuming 500 Hz)
TEMPLATE_LENGTH = TEMPLATE_DURATION * SAMPLE_RATE + 1  # Number of samples per template
CORRELATION_THRESHOLD = 0.4         # Define a threshold for maximum correlation
WINDOW_STEP = 1                     # Step size for the moving window in seconds

###############################################################################
# Read Continuous Data and Event Catalog
# ---------------------------
# Read continuous data paths and templates' folder
data_paths = glob.glob('data/*.mseed')
template_paths = glob.glob('database/*.npy')  # Adjust this path pattern based on your template structure

# Make sure we have continuous data files
if not data_paths:
    raise FileNotFoundError("No continuous data files found in the specified path.")

# Test with the first continuous data file
first_data_file = data_paths[0]  # Take the first continuous data file
print(f"Testing with Continuous Data File: {first_data_file}")

# Read the continuous data stream
st = obspy.read(first_data_file)

# Apply a band-pass filter to the continuous data (1-200 Hz)
st.filter('bandpass', freqmin=1, freqmax=200, corners=4, zerophase=False)

# Get the station name from the first trace
station_name = st[0].stats.station
print(f"Using Station: {station_name}")

# Filter template paths to get only those that match the first station's templates
specific_templates = [tp for tp in template_paths if station_name in tp]

# Ensure we have corresponding templates for the first station
if not specific_templates:
    raise FileNotFoundError(f"No template files found for station: {station_name}")

# Dictionary to store results
results_dict = {}

###############################################################################
# Process Each Continuous Data File
# ---------------------------
# Calculate the number of segments with a moving window of 1 second
num_segments = int(len(st[0].data) / SAMPLE_RATE) - TEMPLATE_DURATION + 1

# Iterate through each segment of the continuous data with a moving window
for i in range(num_segments):
    # Extract a segment of continuous data with the corresponding template length
    start_index = i * SAMPLE_RATE  # Convert seconds to sample index
    segment = st[0].data[start_index:start_index + TEMPLATE_LENGTH]

    # Normalize the segment to have a maximum absolute value of 1
    if np.max(np.abs(segment)) == 0:  # Check to avoid division by zero
        continue
    segment /= np.max(np.abs(segment))

    # Process each template file corresponding to the first station
    for template_file in specific_templates:

        results = []
        # Load the extracted template
        template_data = np.load(template_file)

        # For each template, compute the matched filter
        for template in template_data:
            # Ensure the template has the appropriate size
            if len(template) != TEMPLATE_LENGTH:
                print(f"Template size mismatch for {os.path.basename(template_file)}. Skipping.")
                continue

            # Cross-correlate the normalized segment with the template
            results.append(np.abs(Xcorr(segment, template)))

        # Find the maximum correlation value and its index
        max_corr_index = np.argmax(results)
        max_corr_value = results[max_corr_index]

        # Calculate the start time based on the segment index
        start_time = st[0].stats.starttime + (i * WINDOW_STEP)

        # Check if the maximum correlation exceeds the threshold
        if max_corr_value > CORRELATION_THRESHOLD:
            # Store the information in the results dictionary
            results_dict[start_time.isoformat()] = {
                "station": station_name,
                "max_correlation": max_corr_value,
                "segment_index": i
            }
            
            # Format the start time in a string format suitable for filenames
            start_time_str = start_time.strftime("%Y%m%d_%H%M%S")

            # Save detected event segment as a NumPy file with start time in the filename
            detected_event_filename = os.path.join(detected_events_dir, f"detected_event_{station_name}_{start_time_str}.npy")
            np.save(detected_event_filename, segment)

            # Print the results
            print(f"Template from {os.path.basename(template_file)} at segment {i}:")
            print(f"  Max Correlation: {max_corr_value:.4f} for segment starting at {start_time.isoformat()}")



# Optionally save the results dictionary to a file for further analysis
with open(os.path.join(output_dir, 'matched_filter_results.json'), 'w') as f:
    json.dump(results_dict, f, indent=4)
    
# Print the number of detected events
print(f"\nNumber of detected events: {len(results_dict)}")

###############################################################################
# Plotting Detected Events
# ---------------------------
# Load detected events (.npy files)
detected_events_files = glob.glob(f'{detected_events_dir}/*.npy')

# Select up to 50 samples to plot (or fewer if less available)
num_samples_to_plot = min(50, len(detected_events_files))

# Create a plot for the detected events
plt.figure(figsize=(20, 40))  # Adjust size for better visibility

# Sample rate (assuming 500 Hz)
sample_rate = 500

# Determine the grid size for subplots
num_rows = 10  # 10 rows
num_cols = 5   # 5 columns

for i in range(num_samples_to_plot):
    # Load the detected event
    event_data = np.load(detected_events_files[i])
    
    # Create a time axis for plotting (length of the event data)
    time_axis = np.linspace(0, len(event_data) / sample_rate, num=len(event_data))  # Time from 0 to duration

    # Plot the event data in a subplot
    plt.subplot(num_rows, num_cols, i + 1)  # 10 rows and 5 columns of subplots
    plt.plot(time_axis, event_data, label=f'Detected Event {i + 1}', color='b')
    plt.title(f'Detected Event {i + 1}', fontsize=10)
    plt.xlabel('Time (s)', fontsize=8)
    plt.ylabel('Amplitude', fontsize=8)
    plt.grid()
    plt.legend(fontsize=8)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig('detected_events_plot.png')  # Save the figure as a PNG file
plt.show()

