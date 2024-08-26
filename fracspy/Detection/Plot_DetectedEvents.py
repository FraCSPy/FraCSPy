r"""
Visualization of Detected Events
=================================

Overview
--------
In this section, we extract and visualize the top 50 detected seismic events based on their maximum cross-correlation values. The detected events were previously stored in a results dictionary, which included the start time, station names, and maximum correlation values.

Methodology
-----------
1. **Input Data**: 
   - The script reads the results dictionary generated from the matched filtering process.

2. **Waveform Extraction**: 
   - For each detected event, it extracts a 3-second waveform, capturing 1 second before the detected time and 3 seconds after.

3. **Plotting Configuration**:
   - The waveforms of the detected events will be visualized in a grid of subplots (10 rows and 5 columns).
   - The font size of the labels and titles will be adjusted for better visibility.

4. **Output**:
   - The resulting plot will display the waveforms of the top detected events, saved as **`top_detected_events_plot.png`** for further analysis and reference.
"""

###############################################################################
# Imports
# ---------------------------
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import json
import obspy
import pandas as pd

###############################################################################
# Load Results Dictionary
# ---------------------------
# Load the results dictionary from the JSON file
with open('output/matched_filter_results.json', 'r') as f:
    results_dict = json.load(f)

# Prepare to store only relevant detected events
correlation_data = []

###############################################################################
# Extracting Data for Correlation Analysis
# ---------------------------
# Loop through the detected events in the results dictionary
for detected_time_str, details in results_dict.items():
    correlation_data.append({
        'start_time': detected_time_str,
        'station': details['station'],
        'max_correlation': details['max_correlation']
    })

# Convert to DataFrame for easier manipulation
correlation_df = pd.DataFrame(correlation_data)

# Sort the DataFrame by max_correlation in descending order and take the top 50
top_correlations = correlation_df.sort_values(by='max_correlation', ascending=False).head(50)

# Print the top correlation values for reference
print("Top 50 Highest Correlation Values:")
print(top_correlations)

###############################################################################
# Parameters for Waveform Extraction
# ---------------------------
# Create a list to store the extracted waveforms for plotting
waveform_samples = []

# Parameters
SAMPLE_RATE = 500                  # Assume a sample rate of 500 Hz
WAVEFORM_DURATION = 3               # Duration of each waveform in seconds
PRE_EVENT_DURATION = 0              # 1 second before the detected time
POST_EVENT_DURATION = 3             # 3 seconds after the detected time

###############################################################################
# Iterating through the Top Detected Events
# ---------------------------
# Iterating through the top detected events
for index, row in top_correlations.iterrows():
    detected_time = obspy.UTCDateTime(row['start_time'])
    station_name = row['station']
    
    # Read the corresponding continuous data file for the station
    continuous_data_path = glob.glob(f'data/*{station_name}*.mseed')  # Adjust this pattern accordingly
    if not continuous_data_path:
        print(f"No continuous data file found for station: {station_name}")
        continue
    continuous_data_file = continuous_data_path[0]  # Take the first matched file
    #print(f"Reading continuous data from: {continuous_data_file}")

    # Read the continuous data stream
    st = obspy.read(continuous_data_file)
    st.filter('bandpass', freqmin=1, freqmax=200, corners=4, zerophase=False)

    # Calculate start and end times for waveform extraction
    start_time = detected_time - PRE_EVENT_DURATION
    end_time = detected_time + POST_EVENT_DURATION

    # Trim the stream to extract the specified waveform
    waveform_segment = st.trim(start_time, end_time)

    # Ensure we are getting the first channel (Z-component)
    if len(waveform_segment) > 0:
        data_array = waveform_segment[0].data  # Get the first channel data
        waveform_samples.append((data_array, detected_time.isoformat()))  # Store data with the detected time
    else:
        print(f"Waveform extraction failed for {station_name} at {row['start_time']}")

###############################################################################
# Plotting Detected Events
# ---------------------------
# Font size for the figure
font = {'family': 'normal',
        'weight': 'bold',
        'size': 20}

# Now we plot the highest correlation waveforms
plt.figure(figsize=(30, 40))  # Adjust size for better visibility
plt.rc('font', **font)

# Define plot layout for 50 events: 10 rows and 5 columns
num_rows = 10
num_cols = 5

for i in range(len(waveform_samples)):
    # Load the detected event data and corresponding detected time
    event_data, detected_time_str = waveform_samples[i]

    # Create a time axis for plotting
    time_axis = np.linspace(0, WAVEFORM_DURATION, num=len(event_data))  # Time from 0 to duration

    # Plot the event data in a subplot
    plt.subplot(num_rows, num_cols, i + 1)  # 10 rows and 5 columns of subplots
    plt.plot(time_axis, event_data, label=f'Detected Event {i + 1}', color='b')
    plt.title(f'Detected Event {i + 1}\nTime: {detected_time_str}', fontsize=20, fontweight='bold')  # Larger title font size
    
    # Set x and y labels conditionally
    if i % num_cols == 0:  # First column only
        plt.ylabel('Amplitude', fontsize=20, fontweight='bold')  # Increase font size for y-labels
    else:
        plt.yticks([])  # Remove y-axis ticks for other columns
    if i // num_cols == num_rows - 1:  # Last row only
        plt.xlabel('Time (s)', fontsize=20, fontweight='bold')  # Increase font size for x-labels
    else:
        plt.xticks([])  # Remove x-axis ticks for other rows

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig('top_detected_events_plot.png')  # Save the figure as a PNG file
plt.show()
