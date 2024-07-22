import numpy as np

from fracspy.location.utils import get_max_locs
from fracspy.location.utils import moveout_correction
from fracspy.location.utils import semblance_stack


def diffstack(data, n_xyz, Op, nforhc=10):
    """Kirchhoff migration for microseismic source location

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


def semblancediffstack(data, n_xyz, tt, dt, nforhc=10):
    """Diffraction stacking for microseismic source location

    This routine performs imaging of microseismic data by diffraction
    stacking . In practice,
    this approach is similar to :func:`fracspy.location.migration.diffstack`
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
    nforhc : :obj:`int`, optional
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

    # Loop over grid points
    for igrid in range(ngrid):
        # Get traveltime curve
        tcurve = ttg[:,igrid]
        # Get minimum time
        tmin = np.min(tcurve)
        # Get shifts
        itshifts =  np.round((tcurve - tmin)/dt)
        # Perform moveout correction for data
        data_mc = moveout_correction(data=data,itshifts=itshifts)

        ds_im[igrid] = np.max(semblance_stack(data_mc))

    ds_im_vol = ds_im.reshape(nx, ny, nz)
    
    hc, _ = get_max_locs(ds_im_vol, n_max=nforhc, rem_edge=False)
    return ds_im_vol, hc
