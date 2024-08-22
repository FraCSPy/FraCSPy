import numpy as np


def _get_centroid(array_xyz):
    return np.mean(array_xyz, axis=1)


def get_max_locs(ssimage, n_max=50, rem_edge=True, edgebuf=10, absval=True):

    if absval: ssimage=abs(ssimage)
    if rem_edge:
        if len(ssimage.shape)==2:
            cropped_image = ssimage[edgebuf:-edgebuf, edgebuf:-edgebuf]
        elif len(ssimage.shape)==3:
            cropped_image = ssimage[edgebuf:-edgebuf, edgebuf:-edgebuf,  edgebuf:-edgebuf]
        ev_locs = np.array(np.unravel_index(np.argpartition(cropped_image.ravel(), -1 * n_max)[-n_max:],
                                            cropped_image.shape))

        ev_locs = ev_locs + edgebuf
    else:
        ev_locs = np.array(np.unravel_index(np.argpartition(ssimage.ravel(), -1 * n_max)[-n_max:],
                                         ssimage.shape))

    if n_max > 1:
        ev_loc = _get_centroid(ev_locs)
    else: ev_loc = ev_locs

    return ev_loc, ev_locs


def dist2rec(recs, gx, gy, gz):
    r"""Compute distances from a 3D grid of points to array of receivers.

    Parameters
    ----------
    recs : :obj:`float`
        receiver coordinates [3,nr]
    gx : :obj:`float`
        x coordinates of a grid [1,ngx]
    gy : :obj:`float`
        y coordinates of a grid [1,ngy] 
    gz : :obj:`float`
        z coordinates of a grid [1,ngz]

    Returns
    -------
    d : :obj:`numpy.ndarray`
        4D array of distances [nr,ngx,ngy,ngz]

    """
    nr = recs.shape[1]
    gx, gy, gz = np.meshgrid(gx, gy, gz, indexing='ij')
    d = np.sqrt((recs[0][:, None, None, None] - gx)**2 +
                (recs[1][:, None, None, None] - gy)**2 +
                (recs[2][:, None, None, None] - gz)**2)
    return d


def moveout_correction(data:np.ndarray, itshifts:np.ndarray):
    r"""Moveout correction for microseismic data.

    This function applies a moveout correction to microseismic data by shifting each sample in time according to its corresponding shift value.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        input seismic data [nr, nt]
    itshifts : :obj:`numpy.ndarray`
        array of shift values [nr]

    Returns
    -------
    data_corrected : :obj:`numpy.ndarray`
        microseismic data with corrected moveout [nr, nt]

    Notes
    -----
    The function checks that all values in `itshifts` are non-negative and that the length of `itshifts` matches the number of rows in `data`.

    Example:
    >>> # Assuming you have a 2D array "data" with shape (nr, nt) and an array "itshifts" with shape (nr,)
    >>> corrected_data = moveout_correction(data, itshifts)
    """
    # Get size
    nr, nt = data.shape
    
    # Check size
    if len(itshifts) != nr:
        raise ValueError("The length of itshifts must match the number of rows in data.")
    
    data_corrected = np.zeros_like(data)  # Create an array of zeros with the same shape as data

    # Check if all values in itshifts are positive
    if np.any(itshifts < 0):
        raise ValueError("All values in itshifts must be non-negative.")

    # Calculate the shifted indices for each row
    shifted_indices = np.arange(nt) - itshifts.astype(int)[:, np.newaxis]
    
    # Clip indices to ensure they stay within bounds
    shifted_indices = np.clip(shifted_indices, 0, nt - 1)

    # Copy values from data to result using fancy indexing
    data_corrected[np.arange(nr)[:, np.newaxis], shifted_indices] = data

    return data_corrected

def semblance_stack(data:np.ndarray, semwinsize:int=0):
    r"""Computes the semblance stack for a given input array.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        The input array with shape (nr, nt), where nr is the number of receivers and nt is the number of time samples, usually microseismic data with corrected moveout [nr, nt]
    semwinsize : :obj:`int`, optional, default: 0
        Time window size for semblance, amount of time steps

    Returns
    -------
    semblance_values: :obj:`numpy.ndarray`
        A 1D array with shape (nt,) containing the semblance values for each time sample.

    Description
    -----------
    The semblance_stack function computes the semblance stack, which is a measure of the coherence or similarity between the seismic traces. 
    The semblance value at each time sample is calculated as the ratio of the numerator (the squared sum of amplitudes at each trace) to the denominator 
    (the sum of squares of amplitudes at each trace, multiplied by the number of traces).

    Mathematical Formula
    --------------------
    The formula for computing the semblance stack is:

    semblance = numerator / denominator

    where:

    numerator = (Σ(data, axis=0))^2
    denominator = nr * Σ(data^2, axis=0)
    nr is the number of receivers
    
    Notes
    ----- 
    The function initializes an array semblance_values with shape (nt,) to store the semblance values.
    It then computes the numerator and denominator of the semblance equation using NumPy operations.
    To avoid division by zero, it sets any denominators that are equal to zero to a small value (1e-10).
    Finally, it returns the computed semblance values.
    """
    # Get the shape of the input data, which should be (nr, nt)
    nr, nt = data.shape
    
    # Initialize an array to store the semblance values
    # The size of this array is determined by the number of time samples (nt)
    semblance_values = np.zeros(nt)

    if semwinsize==0:
        # Compute the numerator of the semblance equation
        # This is simply the sum of the squares of each data point
        #numerator = np.sum(data ** 2, axis=0)
        numerator = np.sum(data, axis=0) ** 2

        # Compute the denominator of the semblance equation
        # The denominator is a function of the total amplitude and the square root of the variance
        #sum_data_squared = np.sum(data, axis=0) ** 2
        #denominator = nr * np.sum(data ** 2, axis=0) - sum_data_squared
        denominator = nr * np.sum(data ** 2, axis=0)
        #denominator = nr * np.sum(data, axis=0) ** 2

        # Avoid division by zero
        denominator[denominator == 0] = 1e-10

        # Compute the semblance values using the numerator and denominator
        # The semblance value at each time sample is calculated as the ratio of the numerator to the denominator
        semblance_values = numerator / denominator

    return semblance_values