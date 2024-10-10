
import obspy
import numpy as np
import glob
import os
import json

def Xcorr(x, y):
    """
    Calculate the negative dot product of two normalized vectors.

    This function normalizes the input vectors and computes 
    the negative sum of the product of their corresponding elements, 
    effectively calculating the negative dot product. 

    Parameters
    ----------
    x : np.ndarray
        The first input array (vector) to compare.
    y : np.ndarray 
        The second input array (vector) to compare.

    Returns
    -------
    loss : float 
        The negative dot product of the normalized vectors.
    
    Example:
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> y = np.array([4.0, 5.0, 6.0])
        >>> result = Xcorr(x, y)
        >>> print(result)
        -0.9746318461970762
    """

    # Normalize the input arrays
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    
    # Calculate the negative dot product
    loss = -np.sum(np.multiply(x, y))
    return loss
  
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

# ###############################################################################
#                   Associate Detected Events Across Stations                  
# ---------------------------                                                    
# Read JSON files and group events detected within the same time window.       

def associate_detected_events(results_dict, time_window=4, num_station=3):
    """
    Associates detected events across different stations based on time.

    Parameters
    ----------
    results_dict : list
        List of dictionaries containing event information per station
    time_window : int
        Time window in seconds to consider events as associated (default: 4 seconds).
    num_station : int
        Minimum number of stations required to consider it as associated detection (default: 3).

    Returns
    -------
    associated_events : list
        List of associated events, where each event is a dictionary containing:
            - "time": The start time of the event in UTC format.
            - "stations": List of stations that detected the event.
            - "max_correlation": Maximum correlation value across stations.
    """
    
    events = []
    
    for trace_results_dict in results_dict:
        for event_time_str, event_details in trace_results_dict.items():
            # Convert to UTC time
            time_float = obspy.UTCDateTime(event_time_str).timestamp
            events.append({
                'time': time_float,
                'station': event_details['station'],
                'max_correlation': event_details['max_correlation'],
            })
        
    # Now group events based on the time window
    associated_events = []
    events.sort(key=lambda x: x['time'])  # Sort events by time
    
    while events:
        current_event = events.pop(0)
        current_time = current_event['time']
        current_association = {
            'time': current_time,  # Keep in UTC timestamp
            'stations': {current_event['station']},  # Use a set to ensure unique station names
            'max_correlation': current_event['max_correlation'],
        }

        # Check for subsequent events within the time window
        while events and (events[0]['time'] - current_time <= time_window):
            next_event = events.pop(0)
            current_association['stations'].add(next_event['station'])  # Add station if it's different
            current_association['max_correlation'] = max(current_association['max_correlation'], next_event['max_correlation'])

        # Check if at least two stations detected in the same time window
        if len(current_association['stations']) >= num_station:
            current_association['stations'] = list(current_association['stations'])  # Convert set to list
            # Convert time back to UTC for the result dictionary
            current_association['time'] = obspy.UTCDateTime(current_association['time']).isoformat()  # Keep the time in ISO format
            associated_events.append(current_association)

    return associated_events
