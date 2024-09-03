import numpy as np

from fracspy.location.utils import get_max_locs
from fracspy.location.utils import moveout_correction
from fracspy.location.utils import semblance_stack

def kmigration(data, n_xyz, Op, nforhc=10):
    r"""Kirchhoff migration for microseismic source location.

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

def diffstack(data: np.ndarray,
                       n_xyz: tuple, 
                       tt: np.ndarray, 
                       dt: float,                        
                       nforhc: int = 10, 
                       output_type: str = None,
                       stack_type: str = None,
                       pol_correction: str = None,
                       swsize: int = 0):
    r"""Diffraction stacking for microseismic source location.

    This routine performs imaging of microseismic data by diffraction stacking with 
    various stacking approaches and optional polarity correction.
    
    Diffraction stacking is applied to data with corrected event moveout (EMO),
    and can be done in several ways:
    
    1. Absolute-value based
    
    .. math::
        F(\mathbf{r},t) = \left| \sum_{R=1}^{N_R} A_R^{EMO}(t,\mathbf{r}) \right|,

    where :math:`\mathbf{r}` is a vector that defines a spatial position :math:`(x, y, z)` of the image point, 
    :math:`A_R^{EMO}(t,\mathbf{r})` represents the EMO-corrected data at the receiver :math:`R`, 
    and :math:`N_R` is a number of receivers [1]_, [2]_.

    2. Squared-value based

    Similarly, but the stack is taken to the power of 2:   
    
    .. math::
        E(\mathbf{r},t) = \left[ \sum_{R=1}^{N_R} A_R^{EMO}(t,\mathbf{r}) \right]^2,
    
    3. Semblance-based

    Simple stacking can be further improved using a semblance-based approach.

    The semblance is a coherency or similarity measure and can be understood as the ratio of the total energy (the square of sum of amplitudes) 
    to the energy of individual traces (the sum of squares) [3]_:

    .. math::
        S(\mathbf{r},t) = \frac{\left[ \sum_{R=1}^{N_R} A_R^{EMO}(t,\mathbf{r}) \right]^2}
        {N_R \sum_{R=1}^{N_R} \left[ A_R^{EMO}(t,\mathbf{r}) \right]^2}    

    In order to suppress the effect of noise even better, it is possible to extend the semblance-based approach 
    by introducing a sliding time window :math:`W` over which the energy measures are summed:

    .. math::
        S_W(\mathbf{r},t,W) = \frac{\sum_{k=it-W}^{it+W}\left[ \sum_{R=1}^{N_R} A_R^{EMO}(t,\mathbf{r}) \right]^2}
        {N_R \sum_{k=it-W}^{it+W}\sum_{R=1}^{N_R} \left[ A_R^{EMO}(t,\mathbf{r}) \right]^2}

    where :math:`k` an index of the time-discretised signal within a sliding time interval 
    consisting of the :math:`2W + 1` samples, and :math:`it` is the index of time :math:`t` [4]_.

    Depending on the desired output, the function can output either a 3D volume and a location, 
    or a full 4D array (time dimension + 3 spatial dimensions).
    
    By default, a 3D image volume obtained as a maximum of the 4D image function, e.g.:

    .. math::
        I(\mathbf{r}) = \max_t F(\mathbf{r},t).
    
    Optionally, one can output a 3D image volume obtained as a sum of the 4D image function over time, e.g.:

    .. math::
        I(\mathbf{r}) = \sum_t F(\mathbf{r},t).

    Full 4D output is useful for subsequent detection of multiple events based on diffraction stacking.

    Polarity correction, if requested, is done using moment tensor inversion:

   
    
    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of shape :math:`n_r \times n_t`
    n_xyz : :obj:`tuple`
        Number of grid points in X-, Y-, and Z-axes for the imaging area
    tt : :obj:`numpy.ndarray`
        Traveltime table of size :math:`n_r \times n_x \times n_y \times n_z`
    swsize : :obj:`int`, optional, default: 0
        Sliding time window size for semblance-based type, amount of time steps
    nforhc : :obj:`int`, optional, default: 10
        Number of points for hypocenter
    output_type : :obj:`str`, optional, default: None
        Output type to produce. Default None is the same as "max".
        Types: "max" (output 3D volume as a maximum of the image function for all time moments), 
               "sum" (output 3D volume as a sum of the image function through time), 
               "full" (output full 4D image function: x,y,z,t). in this case hc is set to None
    stack_type : :obj:`str`, optional, default: None
        Diffraction stacking type (imaging condition), default None is the same as "absolute" (absolute value).
        Types: "absolute" (absolute value), "squared" (squared value), "semblance" (semblance-based)
    pol_correction : :obj:`str`, optional, default: None
        Polarity correction to be used for stacked amplitudes. 
        None is default for no polarity correction, "mti" is for polarity correction based on moment tensor inversion.

    Returns
    -------
    ds_im_vol : :obj:`numpy.ndarray`
        Diffraction stack volume, returned if output_type is not "full"
    hc : :obj:`numpy.ndarray`
        Estimated hypocentral location, returned if output_type is not "full"
    ds_full : :obj:`numpy.ndarray`
        Diffraction stack 4D array (x,y,z,t), returned if output_type is "full"

    Raises
    ------
    ValueError :
        if data and grid sizes do not fit the traveltimes array
    ValueError :
        if output_type value is unknown
    ValueError :
        if stack_type value is unknown
    ValueError :
        if pol_correction value is unknown

    References
    ----------
    .. [1] Anikiev, D. (2015). Joint detection, location and source mechanism 
       determination of microseismic events (Doctoral dissertation). 
       St. Petersburg State University. St. Petersburg. 
       https://disser.spbu.ru/files/phdspsu2015/Anikiev_PhD_web_final.pdf

    .. [2] Anikiev, D., Valenta, J., Staněk, F. & Eisner, L. (2014). Joint location and 
       source mechanism inversion of microseismic events: Benchmarking on seismicity 
       induced by hydraulic fracturing. Geophysical Journal International, 198(1), 
       249–258. 
       https://doi.org/10.1093/gji/ggu126
    
    .. [3] Neidell, N. S., & Taner, M. T. (1971). SEMBLANCE AND OTHER COHERENCY MEASURES 
       FOR MULTICHANNEL DATA. Geophysics, 36(3), 482–497. 
       https://doi.org/10.1190/1.1440186

    .. [4] Trojanowski, J., & Eisner, L. (2016). Comparison of migration‐based location 
       and detection methods for microseismic events. Geophysical Prospecting, 65(1), 
       47–63. 
       https://doi.org/10.1111/1365-2478.12366

    """
    # Get sizes
    nx, ny, nz = n_xyz
    ngrid = nx*ny*nz
    nr, nt = data.shape   

    # Check sizes
    if nr != tt.shape[0]:
        raise ValueError(f"Number of traces in data is not consistent with traveltimes: {nr} != {tt.shape[0]}")
    if nx != tt.shape[1]:
        raise ValueError(f"Input nx is not consistent with traveltimes: {nx} != {tt.shape[1]}")
    if ny != tt.shape[2]:
        raise ValueError(f"Input ny is not consistent with traveltimes: {ny} != {tt.shape[2]}")
    if nz != tt.shape[3]:
        raise ValueError(f"Input nz is not consistent with traveltimes: {nz} != {tt.shape[3]}")
    
    # Check output type
    if output_type is None:
        output_type = "max"
    if output_type not in ["max","sum","full"]:
        raise ValueError(f"Output type is unknown: {output_type}")

    # Check stacking type
    if stack_type is None:
        stack_type = "absolute"
    if stack_type not in ["absolute","squared","semblance"]:
        raise ValueError(f"Diffraction stacking type is unknown: {stack_type}")

    # Check polarity correction
    if pol_correction is not None:
        if pol_correction not in ["mti"]:
            raise ValueError(f"Polarity correction type is unknown: {pol_correction}")

    # Reshape tt array
    ttg = tt.reshape(nr, -1)

    # Initialise image arrays
    ds_full = np.zeros((nt,ngrid))
    ds_im = np.zeros(ngrid)

    # Find time sample shifts for all grid points
    itshifts = np.round((ttg - ttg.min(axis=0))/dt)
    
    # Loop over grid points
    for igrid in range(ngrid):
        # Perform moveout correction for data
        data_mc = moveout_correction(data=data,itshifts=itshifts[:,igrid])
        if pol_correction is not None:
            # Perform polarity correction for data
            #data_pc = polarity_correction(data=data,pol_correction)
            data_pc = data_mc
        else:
            data_pc = data_mc
        if stack_type is "absolute":
            ds = np.abs(np.sum(data_pc,axis=0))
        if stack_type is "squared":            
            ds = (np.sum(data_pc,axis=0))**2            
        elif stack_type is "semblance":
            # Perform semblance-based stacking
            ds = semblance_stack(data_pc,swsize)

        # Fill based on the output type
        if output_type is "max":
            ds_im[igrid] = np.max(ds)
        elif output_type is "sum":
            ds_im[igrid] = np.sum(ds)            
        else:
            ds_full[:,igrid] = ds

    # Construct output based on its type
    if output_type in ["max","sum"]:
        ds_im_vol = ds_im.reshape(nx, ny, nz)            
        hc, _ = get_max_locs(ds_im_vol, n_max=nforhc, rem_edge=False)
        # Return 3D array and location
        return ds_im_vol, hc
    else:
        # Return reshaped 4D array
        return ds_full.reshape(nt, nx, ny, nz)
