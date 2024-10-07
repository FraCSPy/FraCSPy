r"""
Associate Detected Events Across Stations
===================================================

This script loads the results of detected seismic events saved in JSON format, associates detected events 
across different stations based on their timestamps, and saves the results containing only events detected 
by at least two stations within a specified time window.


Example Usage
-------------
To run this script, ensure that the output directory containing the JSON files 
from the matched filtering process is correctly specified.
"""
from fracspy.detection.associate_detected_events import *
###############################################################################
#  Main Execution                                
# ---------------------------                                                  
# Example usage (assuming output_dir is defined and populated with JSON files following the matched_filtering process).                            

# Specify the input and output directories containing JSON files with detected events
input_Json_dir = '../data/TOC2ME_Data/output'  #: Adjust as necessary
output_dir = '../data/TOC2ME_Data/'            #: Adjust as necessary

# Call the function to associate detected events
associated_events = associate_detected_events(input_Json_dir,time_window=4,num_station=3)

# Optionally save the associated events to a JSON file
with open(os.path.join(output_dir, 'associated_detected_events.json'), 'w') as f:
    json.dump(associated_events, f, indent=4)

# Print out the associated events for review
print(f"\nNumber of associated detected events: {len(associated_events)}")
