
import obspy
import numpy as np
import glob
import os
import json
from fracspy.detection.Xcorr import *

def matched_filtering(st, template_paths,SAMPLE_RATE,TEMPLATE_DURATION,TEMPLATE_LENGTH,WINDOW_STEP,CORRELATION_THRESHOLD):
    """
    Perform matched filtering on continuous data.

    Parameters
    ----------
    st : obspy.Stream
        The continuous data stream to analyze, typically containing seismic waveform data.

    template_paths : list
        A list of file paths pointing to the template arrays corresponding to the current station. 
        Each template is used to match segments of the continuous data during the filtering process.

    SAMPLE_RATE : int
        The sampling rate of the continuous data stream in Hz. This is essential for calculating 
        the number of samples per template and managing time-window calculations appropriately.

    TEMPLATE_DURATION : int
        The duration of each template in seconds. This specifies how long each template lasts.

    TEMPLATE_LENGTH : int
        The total number of samples per template, calculated based on the template duration 
        and sampling rate. This defines the length of the segment to extract from the 
        continuous data for filtering.

    WINDOW_STEP : int
        The step size (in seconds) for moving through the continuous data stream. This parameter 
        controls how much the analysis window shifts with each iteration while processing segments.

    CORRELATION_THRESHOLD : float
        A threshold value for maximum correlation. If the maximum correlation value of a segment 
        against a template exceeds this threshold, the event is considered detected and stored in 
        the results dictionary.

    Returns
    -------
    results_dict : dict
        A dictionary containing detected events with their corresponding start times, station names, and maximum correlation values.
    """
    
    # Dictionary to store results for the current station
    results_dict = {}

    # Calculate the number of segments with a moving window of 1 second
    num_segments = int(len(st[0].data) / SAMPLE_RATE) - TEMPLATE_DURATION + 1

    # Iterate through each segment of the continuous data with a moving window
    for i in range(num_segments):
        # Extract a segment of continuous data corresponding to the template length
        start_index = i * SAMPLE_RATE  # Convert seconds to sample index
        segment = st[0].data[start_index:start_index + TEMPLATE_LENGTH]

        # Normalize the segment to have a maximum absolute value of 1
        if np.max(np.abs(segment)) == 0:  # Check to avoid division by zero
            continue
        segment /= np.max(np.abs(segment))

        # Process each template file corresponding to the current station
        for template_file in template_paths:

            # Load the extracted template
            template_data = np.load(template_file)

            # For each template, compute the matched filter
            results = []

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
                    "station": st[0].stats.station,
                    "max_correlation": max_corr_value,
                    "segment_index": i
                }

    return results_dict

