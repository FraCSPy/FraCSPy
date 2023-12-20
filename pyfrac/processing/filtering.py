import numpy as np
from scipy.signal import filtfilt, butter


def butter_bandpass(lowcut, highcut, fs, order=5):
    """ Bandpass filter

    Parameters
    ----------
    lowcut: int
        Low cut for bandpass
    highcut: int
        High cut for bandpass
    fs: int
        Sampling frequency
    order: int
        Filter order

    Returns
    -------
        b : np.array
            The numerator coefficient vector of the filter
        a : np.array
            The denominator coefficient vector of the filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """ Apply bandpass filter to trace

    Parameters
    ----------
    data: np.array [1D]
        Data onto which to apply bp filter
    lowcut: int
        Low cut for bandpass
    highcut: int
        High cut for bandpass
    fs: int
        Sampling frequency
    order: int
        Filter order

    Returns
    -------
        y : np.array
            Bandpassed data
    """

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def array_bp(data, lowcut, highcut, fs, order=5):
    """ Apply bandpass filter to array of traces

    Parameters
    ----------
    data: np.array [2D]
        Data onto which to apply bp filter
    lowcut: int
        Low cut for bandpass
    highcut: int
        High cut for bandpass
    fs: int
        Sampling frequency
    order: int
        Filter order

    Returns
    -------
        bp : np.array [2D]
            Bandpassed data
    """
    bp = np.vstack([butter_bandpass_filter(data[:, ix], lowcut, highcut, fs, order)
                    for ix in range(data.shape[1])])

    return bp
