import numpy as np
from scipy.signal import convolve2d
from typing import Union


def _get_centroid(array_xyz):
    """
    Computes the centroid (mean) of a set of points in 3D space.

    This function calculates the centroid of a set of points represented by a 2D numpy array with shape (3, N), where N is the number of points.
    The centroid is the average of all the points along each axis (x, y, z).

    Parameters
    ----------
    array_xyz : numpy.ndarray
        A 2D numpy array of shape (3, N) where each column represents a point in 3D space with coordinates (x, y, z).

    Returns
    -------
    numpy.ndarray
        A 1D numpy array of shape (3,) representing the centroid of the points, with the mean values for each of the x, y, and z coordinates.
    """

    return np.mean(array_xyz, axis=1)


def get_max_locs(ssimage, n_max=50, rem_edge=True, edgebuf=10, absval=True):
    """Source location from image

    Compute the source location from a seismic image.

    Parameters
    ----------
    ssimage : :obj:`numpy.ndarray`
        Image of size :math:`n_x \times n_y \times n_z`
    n_max : :obj:`int`, optional
        Number of maximum values to extract (if ``n_max>1``, the centroid of these values
        will be computed and provided as the estimated source location)
    rem_edge : :obj:`bool`, optional
        Remove edges of volume
    edgebuf : :obj:`int`, optional
        Number of grid points to remove from each edge if ``rem_edge=True``
    absval : :obj:`bool`, optional
        Compute absolute value of ``ssimage``

    Returns
    -------
    ev_loc : :obj:`tuple`
        Most likely source location
    ev_locs : :obj:`tuple`
        `n_max` most likely source locations

    """
    if absval:
        ssimage = np.abs(ssimage)
    if rem_edge:
        if len(ssimage.shape) == 2:
            cropped_image = ssimage[edgebuf:-edgebuf, edgebuf:-edgebuf]
        elif len(ssimage.shape) == 3:
            cropped_image = ssimage[edgebuf:-edgebuf, edgebuf:-edgebuf,  edgebuf:-edgebuf]
        ev_locs = np.array(np.unravel_index(np.argpartition(cropped_image.ravel(), -1 * n_max)[-n_max:],
                                            cropped_image.shape))

        ev_locs = ev_locs + edgebuf
    else:
        ev_locs = np.array(np.unravel_index(np.argpartition(ssimage.ravel(), -1 * n_max)[-n_max:],
                                         ssimage.shape))

    if n_max > 1:
        ev_loc = _get_centroid(ev_locs)
    else:
        ev_loc = ev_locs

    return ev_loc, ev_locs


def get_location_misfit(loca:list, locb:list, steps:list = None):
    r"""
    Calculate the location misfit between two lists of locations (loca and locb) 
    possibly scaled by a list of step values.
    
    Parameters
    ----------
    loca : :obj:`list`
        The first list of locations
    locb : :obj:`list`
        The second list of locations
    steps : :obj:`list`, optional, default: None
        A list of scaling factors. Defaults to None.

    Returns
    -------
        list: The location misfit between the two input lists

    Raises
    ------
    ValueError
        If the input lists have different lengths.

    """
    if len(loca) != len(locb):
        raise ValueError("Input location lists must have the same length.")
    
    if steps is not None and len(steps) != len(loca):
        raise ValueError("Steps list must have the same length as location lists.")
    
    loca_array = np.array(loca)
    locb_array = np.array(locb)
    
    if steps is not None:
        steps_array = np.array(steps)
        return list((loca_array - locb_array) * steps_array)
    else:
        return list(loca_array - locb_array)


def dist2rec(recs, gx, gy, gz):
    r"""Compute distances from a 3D grid of points to array of receivers.

    Parameters
    ----------
    recs : :obj:`numpy.ndarray`
        receiver coordinates of size :math:`3 \times n_r`
    gx : :obj:`numpy.ndarray`
        x coordinates
    gy : :obj:`numpy.ndarray`
        y coordinates
    gz : :obj:`numpy.ndarray`
        z coordinates

    Returns
    -------
    d : :obj:`numpy.ndarray`
        4D array of distances of size :math:`n_r \times n_x \times n_y \times n_z`

    """
    nr = recs.shape[1]
    gx, gy, gz = np.meshgrid(gx, gy, gz, indexing='ij')
    d = np.sqrt((recs[0][:, None, None, None] - gx)**2 +
                (recs[1][:, None, None, None] - gy)**2 +
                (recs[2][:, None, None, None] - gz)**2)
    return d


def moveout_correction(data: np.ndarray, itshifts: np.ndarray):
    r"""Moveout correction for microseismic data.

    This function applies a moveout correction to microseismic data by shifting each sample in time 
    according to its corresponding time index shift value.
    
    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Input seismic data [nr, nt]
    itshifts : :obj:`numpy.ndarray`
        Array of time index shift values [nr]

    Returns
    -------
    data_corrected : :obj:`numpy.ndarray`
        Microseismic data with corrected moveout [nr, nt]

    Notes
    -----
    The function checks that the length of `itshifts` matches the number of rows in `data`.    
    The input `itshifts` array is converted to integer.
    Shifting is done towards the beginning of time for positive `itshifts` values and 
    in the opposite direction for negative.

    Examples
    --------
    >>> # Assuming you have a 2D array "data" with shape (nr, nt) and an array "itshifts" with shape (nr,):
    >>> corrected_data = moveout_correction(data, itshifts)
    >>> # To make reverse correction, apply the negative "itshifts":
    >>> data_restored = moveout_correction(corrected_data, -itshifts)
    """
    # Get size
    nr, nt = data.shape
   
    # Check size
    if len(itshifts) != nr:
        raise ValueError("The length of itshifts must match the number of rows in data.")
      
    # Create an array of zeros with the same shape as data
    data_corrected = np.zeros_like(data)
    
    # Convert shifts to integers
    itshifts_int = itshifts.astype(int)
        
    # Calculate the shifted indices for each row and clip them to ensure they stay within bounds
    shifted_indices = np.clip((np.arange(nt) - itshifts_int[:, np.newaxis]), 0, nt - 1)
    
    # Copy values from data to result using fancy indexing
    data_corrected[np.arange(nr)[:, np.newaxis], shifted_indices] = data

    return data_corrected


def vgtd(x: Union[np.ndarray, float],
         y: Union[np.ndarray, float],
         z: Union[np.ndarray, float],
         recs: np.ndarray) -> np.ndarray:
    r"""
    Compute vectorized Green's tensor derivative for multiple source points.

    Parameters
    ----------
    x : :obj:`numpy.ndarray` or float
        Imaging area grid vector in X-axis or single X coordinate
    y : :obj:`numpy.ndarray` or float
        Imaging area grid vector in Y-axis or single Y coordinate
    z : :obj:`numpy.ndarray` or float
        Imaging area grid vector in Z-axis or single Z coordinate
    recs : :obj:`numpy.ndarray`
        Array of shape (3, nrec) containing receiver coordinates

    Returns
    -------
    g : :obj:`numpy.ndarray` 
        Array of shape (6, nrec, ngrid) containing the Green's tensor derivative for each source point,
        where ngrid is a number of grid points (ngrid=1 if x,y,z are single values)
    """
    # Convert single values to arrays if necessary
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
        
    # Create a meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # Stack the arrays into a (3, ngrid) array
    sources = np.vstack((X.flatten(), Y.flatten(), Z.flatten()))
    
    # Compute relative positions and distances
    rs = recs[:, :, np.newaxis] - sources[:, np.newaxis, :]  # shape (3, nrec, ngrid)
    r2 = np.sum(rs**2, axis=0)  # shape (nrec, ngrid), squared distances
    r = np.sqrt(r2)  # shape (nrec, ngrid), distances
    
    # Normalize and take care of division by zero
    r = np.where(np.isclose(r, 0), np.nan, r)
    rs_norm = rs/r
    rs_norm = np.where(np.isnan(rs_norm), 0, rs_norm)
    # Extract normalized components
    rsx, rsy, rsz = rs_norm[0], rs_norm[1], rs_norm[2]  # each shape (nrec, ngrid)
    
    # Compute Green tensor derivative components
    g = np.empty((6, rsx.shape[0], rsx.shape[1]))  # shape (6, nrec, ngrid)
    g[0] = rsx * rsx * rsz  # shape (nrec, ngrid)
    g[1] = rsy * rsy * rsz  # shape (nrec, ngrid)
    g[2] = rsz * rsz * rsz  # shape (nrec, ngrid)
    g[3] = 2 * rsx * rsy * rsz  # shape (nrec, ngrid)
    g[4] = 2 * rsx * rsz * rsz  # shape (nrec, ngrid)
    g[5] = 2 * rsy * rsz * rsz  # shape (nrec, ngrid)
    
    # Return the Green tensor derivative vector for all grid points
    return np.squeeze(g)


def svd_inv(M: np.ndarray, threshold: float = 1e-15):
    """
    Compute the inverse of a matrix using SVD with regularization.
   
    Parameters:
    ----------
    M : :obj:`numpy.ndarray` 
        The input matrix to invert.
    threshold : :obj:`float` , optional
        Threshold for considering singular values as zero. Default is 1e-15.
       
    Returns:
    -------
    M_inv : :obj:`numpy.ndarray` 
        The regularized inverse of the input matrix.
    """
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Compute regularized inverse singular values
    s_inv = np.zeros_like(s)
    np.divide(1, s, out=s_inv, where=s > threshold)
    
    # Compute the inverse
    return np.dot(Vt.T * s_inv, U.T)


def mgtdinv(g: np.ndarray) -> np.ndarray:
    r"""
    Construct the 6x6 matrix from Green's tensor derivatives and compute its inverse for each grid point.
    
    Parameters
    ----------
    g : :obj:`numpy.ndarray` 
        Array of shape (6, nrec, ngrid) containing the Green's tensor derivative for each source point.
        If size is (6, nrec) it treats it as it is for one single source point.
    
    Returns
    -------
    gtg_inv : :obj:`numpy.ndarray` 
        A set of 6x6 matrices, one for each grid point, of size (6, 6, ngrid) or (6 ,6)
    """    
    # Check number of components
    if g.shape[0]!=6:
        raise ValueError("g must have 6 components!")
    
    # Check dimension and compute
    if g.ndim == 3:
        # Case 1: g is of shape (6, nrec, ngrid)
        
        # Construct gtg as a 6x6 matrix from g for each grid point 
        gtg = np.einsum('imk,jmk->ijk', g, g)  # shape (6, 6, ngrid)
        
        # Inverse of the 6x6 matrix for each grid point        
        #gtg_inv = np.linalg.inv(gtg.transpose(2,0,1)).transpose(1,2,0)

        # Initialize gtg_inv with the same shape as gtg
        gtg_inv = np.zeros_like(gtg)

        # Iterate over each grid point
        for i in range(gtg.shape[2]):
            try:
                # Try inverse of the 6x6 matrix
                gtg_inv[:,:,i] = np.linalg.inv(gtg[:,:,i])
            except np.linalg.LinAlgError:
                # If inversion fails, use SVD with regularization
                gtg_inv[:,:,i] = svd_inv(gtg[:,:,i])
    
    elif g.ndim == 2:
        # Case 2: g is of shape (6, nrec)        
        
        # Compute GTG matrix (single 6x6 matrix)
        gtg = np.dot(g, g.T)  # Shape (6, 6)
        
        # Compute the inverse
        try:
            # Try inverse of the 6x6 matrix
            gtg_inv = np.linalg.inv(gtg)
        except np.linalg.LinAlgError:
            # If inversion fails, use SVD with regularization
            gtg_inv = svd_inv(gtg)
        
    else:
        raise ValueError(f"Invalid shape for g: {g.shape}. Expected 2D or 3D array.")

    # Return the 6x6 inverse matrices for all grid points
    return gtg_inv


def polarity_correction(data: np.ndarray,                         
                        polcor_type: str = "mti",
                        g: np.ndarray = None,
                        gtg_inv: np.ndarray = None):
    r"""Polarity correction for microseismic data with corrected event moveout.

    This function applies a polarity correction to microseismic data with corrected event moveout.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        input seismic data with corrected event moveout [nr, nt]    
    polcor_type : :obj:`str`, optional, default: "mti"
        Polarity correction type to be used for data amplitudes.
    g : :obj:`numpy.ndarray`, optional, default: None
        Array of shape (6, nrec) containing the vectorized Green tensor derivative.
        Required for "mti" polcor_type.
    gtg_inv : :obj:`numpy.ndarray`, optional, default: None 
        6x6 matrix constructed as an inverse from the matrix derived from vectorized Green tensor derivative.
        Required for "mti" polcor_type.

    Returns
    -------
    data_corrected : :obj:`numpy.ndarray`
        microseismic data with corrected polarity [nr, nt]
    
    Notes
    -----
    Polarity correction is done using the moment tensor inversion.

    Raises
    ------
    ValueError :
        if polcor_type value is unknown

    """
    # Get size
    nr, nt = data.shape

    # Check polarity correction type
    if polcor_type not in ["mti"]:
        raise ValueError(f"Polarity correction type is unknown: {polcor_type}")

    # Perform polarity correction based on the type
    if polcor_type=="mti":
        # Check input
        if g is None:
            raise ValueError(f"The g input is required for polarity correction type {polcor_type}")
        if gtg_inv is None:
            raise ValueError(f"The gtg_inv is required for polarity correction type {polcor_type}")
        # Compute the vectorized moment tensor for each time moment
        # M=((G^T*G)^-1)*G^T*A
        mt  = np.dot(gtg_inv, np.dot(g, data))

        # Compute the polarity sign of the dot product between mt and g
        sign_matrix = np.sign(np.dot(mt.T, g).T)
        
        # Apply the polarity correction to the data
        data_corrected = sign_matrix * data

    return data_corrected


def semblance_stack(data:np.ndarray, swsize:int = 0):
    r"""Computes the semblance stack for a given input array.

    The semblance_stack function computes the semblance, which is a measure of the coherence 
    or similarity between the seismic traces. 
    If swsize is 0, the semblance value at each time sample is calculated as the ratio of the numerator 
    (the squared sum of amplitudes at each trace) to the denominator 
    (the sum of squares of amplitudes at each trace, multiplied by the number of traces).

    The formula for computing the semblance is:

    .. math::
        S(it) = \frac{\left[\sum_{R=1}^{N_R} d_R(it)\right]^2}
        {N_R \sum_{R=1}^{N_R} \left[d_R(it)\right]^2}  

    where :math:`N_R` is the number of receivers, and :math:`d_R(it)` is amplitude at trace :math:`R` 
    and time sample :math:`it`.

    A sliding window approach is used when swsize > 0.
    For each time sample, it considers a window of size 2*swsize + 1 centered on that sample.
    The numerator and denominator are computed within this window for each time sample.
    At the edges of the data, where a full window is not available, the function uses as much data as is available.

    The formula for computing the semblance with a sliding window :math:`W` (defined by swsize) is:

    .. math::
        S_W(it) = \frac{\sum_{k=max(0,it-W)}^{min(nt,it+W+1)}\left[\sum_{R=1}^{N_R} d_R(it)\right]^2}
        {N_R \sum_{k=max(0,it-W)}^{min(nt,it+W+1)}\sum_{R=1}^{N_R} \left[d_R(it)\right]^2}

    where :math:`nt` is the maximum number of time samples.
    
    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        The input array with shape (nr, nt), where nr is the number of receivers and nt is the number 
        of time samples, usually microseismic data with corrected moveout [nr, nt]
    swsize : :obj:`int`, optional, default: 0
        Sliding time window size for semblance, time steps

    Returns
    -------
    semblance_values: :obj:`numpy.ndarray`
        A 1D array with shape (nt,) containing the semblance values for each time sample.

    Notes
    ----- 
    The function initializes an array semblance_values with shape (nt,) to store the semblance values.
    It then computes the numerator and denominator of the semblance equation using NumPy and SciPy operations.
    To avoid division by zero, it sets any denominators that are equal to zero to a small value (1e-10).
    Finally, it returns the computed semblance values.

    """
    # Get the shape of the input data, which should be (nr, nt)
    nr, nt = data.shape
    
    # Initialize an array to store the semblance values
    # The size of this array is determined by the number of time samples (nt)
    semblance_values = np.zeros(nt)

    # Compute semblance with or without sliding time window
    if swsize==0:
        # Compute the numerator of the semblance equation                
        numerator = np.sum(data, axis=0) ** 2

        # Compute the denominator of the semblance equation        
        denominator = nr * np.sum(data ** 2, axis=0)

        # Avoid division by zero
        denominator[denominator == 0] = 1e-10

        # Compute the semblance values using the numerator and denominator
        # The semblance value at each time sample is calculated as the ratio of the numerator to the denominator
        semblance_values = numerator / denominator
    elif swsize<0:
        raise ValueError(f"Sliding time window size (swsize) can not be negative:{swsize}")
    else:
        # Simple for-loop implementation (slow)
        # for it in range(nt):
        #     # Determine the start and end of the window
        #     start = int(max(0, it - swsize))
        #     end = int(min(nt, it + swsize + 1))            
            
        #     # Extract the data within the window
        #     window_data = data[:, start:end]
            
        #     # Compute numerator and denominator for this window
        #     numerator = np.sum(np.sum(window_data, axis=0) ** 2)
        #     denominator = nr * np.sum(np.sum(window_data ** 2, axis=0))
            
        #     # Avoid division by zero
        #     if denominator == 0:
        #         denominator = 1e-10
            
        #     # Compute semblance value for this time sample
        #     semblance_values[it] = numerator / denominator
        # Create a window for convolution
        window = np.ones(2 * swsize + 1)
        
        # Compute the sum of data across receivers
        data_sum = np.sum(data, axis=0)
        
        # Compute the squared sum using convolution
        numerator = convolve2d(data_sum[np.newaxis, :] ** 2, window[np.newaxis, :], mode='same')
        
        # Compute the sum of squared data
        data_squared_sum = np.sum(data ** 2, axis=0)
        
        # Compute the denominator using convolution
        denominator = nr * convolve2d(data_squared_sum[np.newaxis, :], window[np.newaxis, :], mode='same')
        
        # Avoid division by zero
        denominator[denominator == 0] = 1e-10
        
        # Compute semblance values
        semblance_values = numerator.flatten() / denominator.flatten()

    return semblance_values


def are_values_close(desired: np.ndarray, actual: np.ndarray, decimal: int = 6) -> np.ndarray:
    """
    Check if the values in two arrays are close to each other up to a specified number of decimal places.

    The function checks whether the absolute difference between `desired` and `actual` 
    is less than 1.5 times 10 raised to the power of `-decimal` for each pair of elements.

    Parameters
    ----------
    desired : :obj:`np.ndarray`
        The desired values to compare.
    actual : :obj:`np.ndarray` 
        The actual values to compare.
    decimal : :obj:`int`, optional
        The number of decimal places to consider for the comparison. Default is 6.

    Returns
    -------
    np.ndarray
        A boolean array where each element is True if the corresponding elements in `desired` 
        and `actual` are close within the specified decimal precision, False otherwise.

    Examples
    --------
    >>> are_values_close(np.array([1.000001, 1.0001]), np.array([1.000002, 1.0002]), decimal=6)
    array([ True, False])
    """
    tolerance = np.float64(1.5 * 10.0**(-decimal))
    return np.abs(desired - actual) < tolerance
