import numpy as np
import matplotlib.pyplot as plt

def plot_station_data(tr_data, station_list, num_stations_to_plot=6, figure_size=(30, 30), font_size=20):
    """
    Plots the normalized seismic data for a given list of stations.

    Parameters:
    tr_data: list of np.ndarray - A list containing the seismic data for each station.
    station_list: list of str - A list of station names corresponding to the data.
    num_stations_to_plot: int - The maximum number of stations to plot.
    figure_size: tuple - A tuple specifying the figure size (width, height).
    font_size: int - The font size for labels and legends.
    """
    
    # Create a new figure for plotting with a specified size for all subplots
    plt.figure(figsize=figure_size)  #: Set the figure size for all subplots

    # Set global font size
    plt.rcParams.update({'font.size': font_size})  # Increase the font size

    # Limit the number of stations to plot
    num_stations_to_plot = min(len(tr_data), num_stations_to_plot)

    # Loop through the specified number of stations
    plot_count= 0 
    for plot_count in range(num_stations_to_plot):
        # Normalize the data to have a maximum absolute value of 1
        data_array = tr_data[plot_count]
        data_array /= np.max(np.abs(data_array))
       
        # Increment subplot index
        plt.subplot(num_stations_to_plot, 1, plot_count + 1)

        # Plot the data for the current station
        plt.plot(data_array, label=f'Station: {station_list[plot_count]}')
        plt.xlabel('Samples', fontsize=font_size)  # Larger font size for x label
        plt.ylabel('Normalized Amplitude', fontsize=font_size)  # Larger font size for y label
        plt.legend(loc="upper right", fontsize=font_size)  # Larger font size for legend
        plt.xlim([0, len(data_array)])
        plot_count+=1
	
    # Show the plot
    plt.tight_layout()  # Ensure that subplots fit into the figure area
    plt.show()



