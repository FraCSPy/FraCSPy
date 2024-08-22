import numpy as np

from fracspy.location.utils import get_max_locs
from fracspy.location.utils import moveout_correction
from fracspy.location.utils import semblance_stack

def diffstack(data, n_xyz, Op, nforhc=10):
    """Kirchhoff migration for microseismic source location.

    This routine performs imaging of microseismic data by migration
    using the adjoint of the Kirchhoff modelling operator.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of shape :math`n_r \times n_t`
    n_xyz : :obj:`tuple`
        Number of grid points in X-, Y-, and Z-axes for the imaging area
    Op : :obj:`pyfrac.modelling.kirchhoff.Kirchhoff`
        Kirchhoff operator
    nforhc : :obj:`int`, optional
        Number of points for hypocenter

    Returns
    -------
    migrated : :obj:`numpy.ndarray`
        Migrated volume
    hc : :obj:`numpy.ndarray`
        Estimated hypocentral location

    """
    nx, ny, nz = n_xyz
    migrated = (Op.H @ data).reshape(nx, ny, nz)
    hc, _ = get_max_locs(migrated, n_max=nforhc, rem_edge=False)
    return migrated, hc

def absdiffstack(data, n_xyz, tt, dt, nforhc=10):
    r"""Absolute diffraction stacking for microseismic source location.
    
    This function applies event moveout correction to the input data and then computes 
    the absolute value of the stacking. The result is a 3D image showing the 
    amplitude of the stacked seismic data.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of shape :math`n_r \times n_t`
    n_xyz : :obj:`tuple`
        Number of grid points in X-, Y-, and Z-axes for the imaging area
    tt : :obj:`numpy.ndarray`
        Traveltime table of size :math`n_r \times n_x \times n_y \times n_z`    
    nforhc : :obj:`int`, optional, default: 10
        Number of points for hypocenter

    Description
    -----------
    The stacking is performed for all possible origin times :math:`t`:   
        
    .. math::
            F(\mathbf{r},t) = \left|\sum_{R=1}^{N_R} A_R 
            \left(t + T_R(\mathbf{r})\right)\right|,

    where :math:`\mathbf{r}` is a vector that defines a spatial position 
    :math:`(x, y, z)` of the image point, :math:`T_R(\mathbf{r})` is the P-wave 
    traveltime from the image point :math:`\mathbf{r}` to a receiver :math:`R`,
    :math:`N_R` is a number of receivers, and :math:`A_R` is the observed waveform 
    at the receiver :math:`R`.

    The term 

    .. math::
            A_R^{EMO}(t,\mathbf{r}) = A_R \left(t + T_R(\mathbf{r})\right)

    represents the data with the corrected event moveout.

    Returns
    -------
    ds_im_vol : :obj:`numpy.ndarray`
        Diffraction stack volume
    hc : :obj:`numpy.ndarray`
        Estimated hypocentral location
    """
    # Get sizes
    nx, ny, nz = n_xyz
    ngrid = nx*ny*nz
    nr = tt.shape[0]

    # Reshape tt array
    ttg = tt.reshape(nr, -1)

    # Initialise ds image array
    ds_im = np.zeros(ngrid)

    # Find time sample shifts for all grid points
    itshifts = np.round((ttg - ttg.min(axis=0))/dt)
    
    # Loop over grid points
    for igrid in range(ngrid):
        # Perform moveout correction for data
        data_mc = moveout_correction(data=data,itshifts=itshifts[:,igrid])
        # Compute absolute value of the stacking
        ds_im[igrid] = np.max(np.abs(np.sum(data_mc,axis=0)))

    ds_im_vol = ds_im.reshape(nx, ny, nz)
    
    hc, _ = get_max_locs(ds_im_vol, n_max=nforhc, rem_edge=False)
    return ds_im_vol, hc

def semblancediffstack(data, n_xyz, tt, dt, semwinsize=0, nforhc=10):
    """Diffraction stacking for microseismic source location based on semblance.

    This routine performs imaging of microseismic data by diffraction
    stacking. In practice, this approach is similar to 
    :func:`fracspy.location.migration.diffstack` and
    :func:`fracspy.location.migration.absdiffstack`
    with the main difference that semblance is used as measure of coherency
    instead of a straight summation of the contributions over the moveout
    curves

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of shape :math`n_r \times n_t`
    n_xyz : :obj:`tuple`
        Number of grid points in X-, Y-, and Z-axes for the imaging area
    tt : :obj:`numpy.ndarray`
        Traveltime table of size :math`n_r \times n_x \times n_y \times n_z`
    semwinsize : :obj:`int`, optional, default: 0
        Time window size for semblance, amount of time steps
    nforhc : :obj:`int`, optional, default: 10
        Number of points for hypocenter

    Returns
    -------
    ds_im_vol : :obj:`numpy.ndarray`
        Diffraction stack volume
    hc : :obj:`numpy.ndarray`
        Estimated hypocentral location
    """
    # Get sizes
    nx, ny, nz = n_xyz
    ngrid = nx*ny*nz
    nr = tt.shape[0]

    # Reshape tt array
    ttg = tt.reshape(nr, -1)

    # Initialise ds image array
    ds_im = np.zeros(ngrid)

    # Find time sample shifts for all grid points
    itshifts = np.round((ttg - ttg.min(axis=0))/dt)
    
    # Loop over grid points
    for igrid in range(ngrid):
        # Perform moveout correction for data
        data_mc = moveout_correction(data=data,itshifts=itshifts[:,igrid])
        # Perform semblance-based stacking
        ds_im[igrid] = np.max(semblance_stack(data_mc),semwinsize)

    ds_im_vol = ds_im.reshape(nx, ny, nz)
    
    hc, _ = get_max_locs(ds_im_vol, n_max=nforhc, rem_edge=False)
    return ds_im_vol, hc
