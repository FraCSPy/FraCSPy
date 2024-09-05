import numpy as np
from scipy.signal import convolve2d

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

    Examples
    --------
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

def vgtd(x: np.ndarray, y: np.ndarray, z: np.ndarray, recs: np.ndarray) -> np.ndarray:
    r"""
    Compute vectorized Green's tensor derivative for multiple source points.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Imaging area grid vector in X-axis
    y : :obj:`numpy.ndarray`
        Imaging area grid vector in Y-axis
    z : :obj:`numpy.ndarray`
        Imaging area grid vector in Z-axis
    recs : :obj:`numpy.ndarray`
        Array of shape (3, nrec) containing receiver coordinates

    Returns
    -------
    g : :obj:`numpy.ndarray` 
        Array of shape (6, nrec, ngrid) containing the Green's tensor derivative for each source point
    """
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
    return g

def svd_inv(M: np.ndarray):
    """
    Compute the inverse of a matrix using SVD with regularization.
    
    Parameters:
    ----------
    M : :obj:`numpy.ndarray` 
        The input matrix to invert.
        
    Returns:
    -------
    :obj:`numpy.ndarray`
        The regularized inverse of the input matrix.
    """
    U, s, Vt = np.linalg.svd(M)

    # Determine inverse singular values
    s = np.where(np.isclose(s, 0), np.nan, s)
    s_inv = 1/s
    s_inv = np.where(np.isnan(s_inv), 0, s_inv)    
    return (Vt.T * s_inv) @ U.T

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


# def vmt(data, gtgi, g):
#     r"""Compute the vectorized moment tensor.

#     From the amplitudes of the input data with corrected event moveout,
#     this function derives the moment tensor using 
#     the given vectorized Green's tensor and the inverse matrix based on it.
#     The moment tensor is computed for each time moment of data.

#     Parameters
#     ----------
#     data : :obj:`numpy.ndarray`
#         The data with corrected event moveout.
#         Shape must be (nrec,nt) where nrec is the number of receivers 
#         and nt is the number of time steps.
#     gtgi : :obj:`numpy.ndarray`
#         The inverse of the Green's function tensor, flattened to 1D.
#         Shape must be (6, 6).
#     g : :obj:`numpy.ndarray`
#         The Green's function tensor, flattened to 1D.
#         Shape must be (6, nrec) where nrec is the number of receivers
#         and m is the number of components.
    
#     Returns
#     -------
#     mt : :obj:`numpy.ndarray`
#         The computed vectorized moment tensor.
#         Shape is (6,nt) where 6 is the number of components 
#         and nt is the number of time steps.

#     """
#     nrec,nt = data.shape
#     ncomp = 6
    
#     mt = np.zeros((6,nt))

#     for it in range(nt):
#         # Compute d using matrix multiplication
#         d = np.dot(g, data[:,it])
            
#         # Compute M using matrix multiplication
#         mt[:,it] = np.dot(gtgi, d)
        
#     return np.dot(gtgi, np.dot(g, data))


def polarity_correction(data: np.ndarray,                         
                        polcor_type: str="mti",
                        g: np.ndarray = None,
                        gtg_inv: np.ndarray = None):
    r"""Polarity correction for microseismic data with corrected event moveout.

    This function applies a polarity correction to microseismic data with corrected event moveout.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        input seismic data with corrected event moveout [nr, nt]
    x : :obj:`float`
        X coordinate of the potential source
    y : :obj:`float`
        Y coordinate of the potential source
    z : :obj:`float`
        Z coordinate of the potential source    
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

def semblance_stack(data:np.ndarray, swsize:int=0):
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