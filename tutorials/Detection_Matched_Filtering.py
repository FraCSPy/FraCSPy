r"""
Matched Filtering for Event Template Detection (Preparing Templates)
=================================================

Overview
--------
This part of the code is designed to prepare templates for seismic event detection. It accomplishes this by reading event times from an event catalog and extracting the corresponding data from continuous seismic recordings. The primary aim is to save these extracted templates for each station, enabling their use in subsequent matched filtering analysis.

Methodology
-----------
- **Data Preparation**:
    - The script reads continuous waveform data, that should be downloaded using the script in "ToC2ME_WorkedExamples" directory.
/preparation", and extracts templates based on the seismic catalog.
    - A band-pass filter is applied to the continuous data, targeting a frequency range of 1-200 Hz.
    - Event times from the catalog are utilized to extract corresponding templates, which are saved for each station.
    - Each extracted template is normalized to ensure a maximum absolute value of 1.

- **Output**:
    - Each template is stored as a NumPy file in a specified output directory.
"""


import obspy
import numpy as np
import glob
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import requests
from fracspy.detection.matched_filtering import *
from fracspy.detection.associate_detected_events import *
###############################################################################
# Constants
# ---------------------------
NUMBER_OF_TEMPLATES = 10  #: Number of templates to extract
TEMPLATE_DURATION = 4      #: Duration of each template (in seconds)
# Specify the month and day you want to work on
target_month = 11  # Example: November
target_day = 1    # Example: 22nd
###############################################################################
# Read Continuous Data and Event Catalog
# ---------------------------
# Create a directory for the matched templates and to save the templates if it doesn't exist.
matched_dir = f"../data/TOC2ME_Data/Matched_Output/"
output_dir = f"../data/TOC2ME_Data/Matched_Output/database/"
os.makedirs(matched_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Load the short event catalog from an Excel file.
# Provide URL for the catalogue
catalogue_url = 'https://zenodo.org/records/6826326/files/ToC2ME_catalog.xlsx?download=1'

# Download the file content directly into a DataFrame
response = requests.get(catalogue_url)

# Check if the request was successful
if response.status_code == 200:
    # Load the Excel data directly from the response content
    df = pd.read_excel(pd.io.excel.ExcelFile(response.content))
    
    print("Data was successfully downloaded and loaded into the DataFrame.")
else:
    print("Failed to download the file. Status code:", response.status_code)
    exit()  # Exit if the download failed



# Filter the DataFrame based on the specified month and day
catalog = df[(df['Month'] == target_month) & (df['Day'] == target_day)]

# Save the filtered data to a new CSV file named 'Catalog_oneDay_Zhang2018_JGR.csv'
catalog.to_csv(os.path.join(matched_dir,'Catalog_oneDay_Zhang2018_JGR.csv'), index=False)

print("Filtered data has been saved to 'Catalog_oneDay_Zhang2018_JGR.csv'")



# Path for the continous data
data_paths = glob.glob('../data/TOC2ME_Data/*.mseed')

# Select the largest 20 magnitude events from the catalog.
catalog = catalog.sort_values(by=['Magnitude'], ascending=False).head(NUMBER_OF_TEMPLATES)

        
###############################################################################
# Process Each Continuous Data File
# ---------------------------
# Loop over each continuous data file.
for data_file in data_paths:  # Adjust to iterate over multiple files if necessary
    # Initialize a list to store station templates.
    station_templates = []
    
    # Print the current data file being processed and the number of events to process.
    print(f"Processing Data File: {data_file}, Number of Events to Process: {len(catalog)}")

    # Loop through the selected events in the catalog.
    for index in range(len(catalog)):
        # Read the continuous data stream.
        st = obspy.read(data_file)
        
        # Apply a band-pass filter to the data (1-200 Hz).
        st.filter('bandpass', freqmin=1, freqmax=200, corners=4, zerophase=False)
        
        # Extract the station name from the stream.
        station_name = st[0].stats.station

        # Create the extraction time based on the event details in the catalog.
        event_time_str = f"2016-{int(catalog.iloc[index]['Month']):02d}-{int(catalog.iloc[index]['Day']):02d}T" \
                         f"{int(catalog.iloc[index]['Hour']):02d}:{int(catalog.iloc[index]['Minute']):02d}:" \
                         f"{int(catalog.iloc[index]['Second']):02d}.000000Z"
        event_time = obspy.UTCDateTime(event_time_str)

        # Formulate the file name for storing the templates.
        file_name = f"EV{int(catalog.iloc[index]['Month']):02d}{int(catalog.iloc[index]['Day']):02d}_" \
                    f"{int(catalog.iloc[index]['Hour']):02d}{int(catalog.iloc[index]['Minute']):02d}" \
                    f"{int(catalog.iloc[index]['Second']):02d}"

        # Trim the stream to the specified time window for the template.
        st_trimmed = st.trim(event_time, event_time + TEMPLATE_DURATION)

        # Extract data from the first channel of the stream.
        data_array = st_trimmed[0].data
        
        # Normalize the data to have a maximum absolute value of 1.
        data_array /= np.max(np.abs(data_array))
        
        # Append the station data to the list of templates.
        station_templates.append(data_array)

    # Save the array of station templates to a .npy file.
    np.save(os.path.join(output_dir, station_name), station_templates)

###############################################################################
# Plotting
# ---------------------------
# After saving the templates, you can visualize one of them
if station_templates:
    # Select one template to plot; here we take the first one as an example
    example_template = station_templates[0]

    # Generate a time axis for the template
    time_axis = np.linspace(0, TEMPLATE_DURATION, len(example_template))

    # Plotting the example template
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, example_template, label='Event Template', color='b')
    plt.title(f'Seismic Event Template for Station: {station_name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Amplitude')
    plt.grid()
    plt.legend()
    plt.show()
else:
    print("No templates were generated.")
    
    


###############################################################################
# Matched Filtering for Event Template Detection (Cross-correlation)
# ---------------------------
# Overview
# --------
# This part of the code implements a matched filtering technique to detect seismic events by comparing extracted templates from various stations against continuous seismic data. The goal is to identify strong correlations between the templates and segments of the continuous data.
# ---------------------------
# Methodology
# -----------
# 1. **Data Preparation**:
#     - The script reads continuous waveform data and the templates.
#    - A band-pass filter is applied to the continuous data, with a frequency range of 1-200 Hz.
# 2. **Template Matching**:
#     - The continuous data is divided into segments that match the length of the templates.
#     - For each segment, the script performs cross-correlation with each template to find the  maximum correlation value.
#     - The start time of the segment is calculated to provide precise temporal information regarding the detected events.
# 3. **Thresholding**:
#    - If the maximum correlation value exceeds a predefined threshold, the results (start time, station name, and correlation value) are stored in a dictionary.
# 4. **Output**:
#     - Each template is saved as a NumPy file in a specified output directory.
#     - A JSON file containing matched filter results for each station is saved.
# ---------------------------
# Key Parameters
# --------------
# - **TEMPLATE_DURATION**: Duration of each template in seconds (default: 4 seconds).
# - **SAMPLE_RATE**: The sampling rate of the continuous data (default: 500 Hz).
# - **CORRELATION_THRESHOLD**: Threshold value for maximum correlation to filter significant results (default: 0.3).

# Output Directory                               
# ---------------------------                                                    
# Create an output directory for results if it doesn't exist                  
output_dir = '../data/TOC2ME_Data/Matched_Output/output'
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
template_paths = glob.glob('../data/TOC2ME_Data/Matched_Output/database/*.npy')  #: Load template files

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
    

###############################################################################
# Plotting Detected Events
# ---------------------------

# After the matched filtering, plot one of the detected events if any events are detected.
if results_dict:  # Check if there are detected events
    # Select the first detected event for demonstration
    first_event_time = list(results_dict.keys())[0]  # Using the first event's start time
    event_info = results_dict[first_event_time]

    # Calculate the start and end time for the event
    event_start_time = obspy.UTCDateTime(first_event_time)
    event_end_time = event_start_time + TEMPLATE_DURATION
    
    # Extract the segment of continuous data corresponding to the detected event
    event_segment = st.trim(event_start_time, event_end_time)

    # Normalizing the segment for better visualization
    event_data_array = event_segment[0].data
    event_data_array /= np.max(np.abs(event_data_array))  # Normalize to max absolute value of 1

    # Create a time axis for plotting
    time_axis = np.linspace(0, TEMPLATE_DURATION, len(event_data_array))

    # Plotting the detected event and its correlation value
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, event_data_array, label='Detected Event', color='g')
    plt.title(f'Detected Event at {station_name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.grid()
    plt.show()
else:
    print(f"No events detected for {station_name}.")


###############################################################################
# Associate Detected Events Across Stations
# ---------------------------
# This part of the code loads the results of detected seismic events saved in JSON format, associates detected events across different stations based on their timestamps, and saves the results containing only events detected  by at least two stations within a specified time window.

# Specify the input and output directories containing JSON files with detected events
input_Json_dir = '../data/TOC2ME_Data/Matched_Output/output'  #: Adjust as necessary
output_dir = '../data/TOC2ME_Data/Matched_Output'            #: Adjust as necessary

# Call the function to associate detected events
associated_events = associate_detected_events(input_Json_dir,time_window=4,num_station=3)

# Optionally save the associated events to a JSON file
with open(os.path.join(output_dir, 'associated_detected_events.json'), 'w') as f:
    json.dump(associated_events, f, indent=4)

# Print out the associated events for review
print(f"\nNumber of associated detected events: {len(associated_events)}")
