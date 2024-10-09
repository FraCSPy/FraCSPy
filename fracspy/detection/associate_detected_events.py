import json
import glob
import obspy
import os
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