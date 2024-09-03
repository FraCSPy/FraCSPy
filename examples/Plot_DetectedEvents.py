r"""
Load and Plot Detected Events Across Stations
========================================================

This script loads the results of associated detected seismic events saved in JSON format,
retrieves the corresponding data for each detected station, and plots examples of 4-second 
segments starting from the detected times.

Example Usage
-------------
To run this script, ensure that the output directory containing the associated detected 
events JSON file is correctly specified and populated with the necessary data.
"""

import json
import os
import glob
import obspy
import matplotlib.pyplot as plt
import random  # Import the random module for random selection

###############################################################################
# Load Associated Detected Events                    
# ---------------------------                                                    
# Load associated detected events from the JSON file.                       

output_dir = '../data/TOC2ME_Data/'  #: Directory containing associated detected events JSON file
save_figures_dir = '../data/TOC2ME_Data/DetectedEventsExample/'  #: Directory to save figures
# Create a directory for saving detected event examples if it doesn't exist
os.makedirs(save_figures_dir, exist_ok=True)

# Load associated detected events
with open(os.path.join(output_dir, 'associated_detected_events.json'), 'r') as f:
    associated_events = json.load(f)

###############################################################################
# Read Continuous Data Paths                          
# ---------------------------                                                    
# Read paths for continuous data files (mseed) and prepare for data access.

#data_paths = glob.glob('../data/TOC2ME_Data/*.mseed')  #: Load continuous data files
data_paths = '../data/TOC2ME_Data/'  #: Load continuous data files

# Ensure that we have continuous data files available
if not data_paths:
    raise FileNotFoundError("No continuous data files found in the specified path.")

###############################################################################
# Randomly Select Detected Events                      
# ---------------------------                                                    
# Select a random subset of events from the associated detected events.       

num_examples_to_plot = 10  #: Number of random events to plot

# Randomly select events without replacement
if len(associated_events) > num_examples_to_plot:
    selected_events = random.sample(associated_events, num_examples_to_plot)
else:
    selected_events = associated_events  #: If not enough events, use all

    
###############################################################################
# Plot Detected Events                                 
# ---------------------------                                                    
# Iterate through associated detected events and plot the data segments.       


# Loop through a limited range of associated events
for event in selected_events:
    # Initialize counter for the number of subfigures
    plot_count = 0

    # Extract relevant event information
    stations = event['stations']  #: List of stations associated with the current event
    start_time = obspy.UTCDateTime(event['time'])  #: The detected event time in UTC

    # Create a new figure for plotting
    plt.figure(figsize=(30, 30))  #: Set the figure size for all subplots

    # Limit the number of stations to plot to a maximum of 10
    num_stations_to_plot = min(len(stations), 10)
    # Loop through the specified number of stations
    for station in stations[:num_stations_to_plot]:
        try:
            # Read the continuous data for the current station
            station_file_path = f"{data_paths}/{station}.5B.mseed"
            st = obspy.read(station_file_path)
            # Apply a band-pass filter to the data (1-200 Hz).
            st = st.filter('bandpass', freqmin=1, freqmax=200, corners=4, zerophase=False)

            # Trim the data to the 4-second segment starting from start_time
            st_trimmed = st.trim(start_time, start_time + 4)  #: Get 4 seconds of data
            data_array = st_trimmed[0].data  #: Extract data from the first trace
            # Normalize the data to have a maximum absolute value of 1.
            #data_array /= np.max(np.abs(data_array))
        
            # Increment subplot index
            plt.subplot(num_stations_to_plot, 1, plot_count + 1)

            # Plot the data for the current station
            plt.plot(data_array, label=f'Station: {station}')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.legend(loc="upper right")

            # Update plot count
            plot_count += 1
        
        except Exception as e:
            print(f"Error reading data for station '{station}': {e}")

    # Improve layout and save the figure to the designated directory
    plt.tight_layout()  #: Adjust subplots to fit into the figure area
    plt.savefig(os.path.join(save_figures_dir, f'detected_event_{start_time.isoformat()}.png'))

    # Clear the current figure to prepare for the next plot
    plt.clf()  #: Clear figure after saving
