import numpy as np

def add_noise(data:np.ndarray, noise_type:str="white", snr:float=1, trind:int=None, seed: int = None):
    r"""Contaminate seismic data with noise of different type.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Input seismic data of shape :math:`n_r \times n_t`
    noise_type: :obj:`str`, optional, default: "white"
        Type of noise: 
        "white" for random white noise, 
        "spiky" for noise that manifests as sharp spikes in the data.
        "ringy" for noise that  creates a ringing effect in the data.
    snr: :obj:`float`, optional, default: 1
        Signal-to-noise ratio to determine noise strength defined 
        as maximum amplitude of the signal divided by the maximum amplitude of noise
    trind: :obj:`int`, optional, default: None (add to all traces)
        Array of indices of traces to which noise must be added 
        (must be >= 0 and <= nr and non-repeating)
    seed: :obj:`int`, optional, default: None
        Seed for the random number generator to ensure reproducibility
    
    Returns
    -------
    data_contaminated : :obj:`numpy.ndarray`
        Data contaminated with noise of the selected noise type, size: :math:`n_r \times n_t`

    Raises
    ------
    ValueError
        If `trind` contains indices that are out of the valid range 
        (>= 0 and < nr) or if `trind` contains non-unique values.

    Notes
    -----
    Maximum amplitude of the signal is calculated as maximum amplitude of the input data.
    
    The ringing effect in seismic data refers to a specific type of noise or distortion that appears as a series of oscillations or reverberations in the seismic trace. This effect is characterized by a repetitive, wave-like pattern that continues after the main seismic event, resembling the ringing of a bell.
    Key aspects of the ringing effect include:

    - Appearance: It looks like a series of alternating positive and negative amplitudes that gradually decrease over time.
    - Causes: Ringing can be caused by various factors, including:
      - Instrument response: Poor coupling between the seismometer and the ground
      - Resonance in the recording system
      - Near-surface reverberations
      - Data processing artifacts, particularly from improper filtering
    - Impact: Ringing can obscure real seismic events and make interpretation difficult, especially for later arrivals or subtle features.
    - Frequency: The ringing often occurs at a characteristic frequency, which can help in identifying its source.
    - Duration: It can persist for a significant portion of the trace, sometimes lasting longer than the actual seismic signal of interest.

    """
    # Set the random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Get the shape of the input data, which should be (nr, nt)
    nr, nt = data.shape

    # Calculate the noise max based on SNR    
    noise_max = np.max(data) / snr

    # Assign noise array
    noise = np.zeros_like(data) 

    # Check trace indices
    if trind is None:
        trind = np.arange(0, nr).astype(int)
    else:
        # Ensure trind contains unique, valid indices
        trind = np.unique(trind)  # Remove duplicates
        if np.any(trind < 0) or np.any(trind >= nr):
            raise ValueError("All indices in trind must be >= 0 and < nr")
        
    if noise_type == "white":
        # Generate white noise
        noise[trind,:] = np.random.normal(0, noise_max, (len(trind), data.shape[1]))
    elif noise_type == "spiky":
        # Generate spiky noise
        for trace in trind:
            num_spikes = np.random.randint(1, 5)  # Random number of spikes
            spike_positions = np.random.choice(nt, num_spikes, replace=False)
            noise[trace, spike_positions] = noise_max * np.random.uniform(-1, 1, num_spikes)
    
    elif noise_type == "ringy":
        # Generate ringy noise
        frequency = np.random.uniform(5, 15)  # Random frequency for the ringing effect
        for trace in trind:
            t = np.arange(nt)
            ringy_wave = noise_max * np.exp(-t/nt) * np.sin(2 * np.pi * frequency * t / nt)
            noise[trace, :] = ringy_wave
    else:
        raise ValueError(f"Invalid noise type: {noise_type}")
    
    # Add noise to the data
    data_contaminated = data + noise

    return data_contaminated