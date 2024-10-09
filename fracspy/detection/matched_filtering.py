
import obspy
import numpy as np
import glob
import os
import json
from fracspy.detection.Xcorr import *

def matched_filtering(trace, 
                      trace_templates, 
                      dt,
                      st_ID,
                      t0,
                      TEMPLATE_DURATION,
                      TEMPLATE_LENGTH,
                      WINDOW_STEP,
                      CORRELATION_THRESHOLD):
    """
    Perform matched filtering on continuous data.

    Parameters
    ----------
    trace : obspy.Stream
        The continuous data stream to analyze, typically containing seismic waveform data.

    trace_templates : list
        A list of file paths pointing to the template arrays corresponding to the current station. 
        Each template is used to match segments of the continuous data during the filtering process.

    dt : int
        The sampling rate of the continuous data stream in Hz. This is essential for calculating 
        the number of samples per template and managing time-window calculations appropriately.
        
    st_ID : str
        Station Identifier, can be a station name, number, etc.

    t0 : UTCDateTime
        start time of trace data

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
    num_segments = int(len(trace) / (1/dt)) - TEMPLATE_DURATION + 1

    # Iterate through each segment of the continuous data with a moving window
    for i in range(num_segments):
        # Extract a segment of continuous data corresponding to the template length
        start_index = i * int(1/dt)  # Convert seconds to sample index
        segment = trace[start_index:start_index + TEMPLATE_LENGTH]

        # Normalize the segment to have a maximum absolute value of 1
        if np.max(np.abs(segment)) == 0:  # Check to avoid division by zero
            continue
        segment /= np.max(np.abs(segment))

        # For each template, compute the matched filter
        results = []
        
        # Process each template file corresponding to the current station
        for template in trace_templates:

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
            start_time = t0 + (i * WINDOW_STEP)

            # Check if the maximum correlation exceeds the threshold
            if max_corr_value > CORRELATION_THRESHOLD:
                # Store the information in the results dictionary
                results_dict[start_time.isoformat()] = {
                    "station": st_ID,
                    "max_correlation": max_corr_value,
                    "segment_index": i
                }

    return results_dict

