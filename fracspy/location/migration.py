import numpy as np

from fracspy.location.utils import *


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
              x: np.ndarray,
              y: np.ndarray,
              z: np.ndarray,
              tt: np.ndarray, 
              dt: float,                        
              nforhc: int = 10, 
              output_type: str = None,
              stack_type: str = None,              
              swsize: int = 0,
              polcor_type: str = None,
              recs: np.ndarray=None):
    r"""Diffraction stacking for microseismic source location.

    This routine performs imaging of microseismic data by diffraction stacking with 
    various stacking approaches and optional polarity correction.
    
    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of shape :math:`n_r \times n_t`
    n_xyz : :obj:`tuple`
        Number of grid points in X-, Y-, and Z-axes for the imaging area
    x : :obj:`numpy.ndarray`
        Imaging area grid vector in X-axis
    y : :obj:`numpy.ndarray`
        Imaging area grid vector in Y-axis
    z : :obj:`numpy.ndarray`
        Imaging area grid vector in Z-axis
    tt : :obj:`numpy.ndarray`
        Traveltime table of size :math:`n_r \times n_x \times n_y \times n_z`    
    dt : :obj:`float`
        Time step
    nforhc : :obj:`int`, optional, default: 10
        Number of points for hypocenter
    output_type : :obj:`str`, optional, default: None
        Defines the output type. In the default case of None it is set to ``max``.
        Types: ``max`` (output 3D volume as a maximum of the image function through time),
        ``mean`` (output 3D volume as an average of the image function through time),                
        ``sumsq`` (output 3D volume as a stack of the squared image function values through time),
        ``full`` (output full 4D image function: x,y,z,t). In this case hc is set to None.
    stack_type : :obj:`str`, optional, default: None
        Diffraction stacking type (imaging condition), default None is the same as ``squared``.
        Types: ``absolute`` (absolute-value), ``squared`` (squared-value), ``semblance`` (semblance-based).
    swsize : :obj:`int`, optional, default: 0
        Sliding time window size for semblance-based type, amount of time steps
    polcor_type : :obj:`str`, optional, default: None
        Polarity correction type to be used for amplitudes.
        None is default: no polarity correction, ``mti`` is for polarity correction based on moment tensor inversion.
    recs : :obj:`numpy.ndarray`, optional, default: None
        Array of shape (3, nrec) containing receiver coordinates.
        Must be provided if polcor_type is not None

    Returns
    -------
    ds_im_vol : :obj:`numpy.ndarray`
        Diffraction stack volume, returned if output_type is not "full"
    hc : :obj:`numpy.ndarray`
        Estimated hypocentral location, if output_type is not "full" is set to None
    ds_full : :obj:`numpy.ndarray`
        Diffraction stack full array of shape (nt, n_xyz), returned if output_type is "full"

    Raises
    ------
    ValueError :
        if data and grid sizes do not fit the traveltimes array
    ValueError :
        if output_type value is unknown
    ValueError :
        if stack_type value is unknown
    ValueError :
        if polcor_type value is unknown
    ValueError :
        if recs is None and polcor_type is not None

    Notes
    -----
    The subsurface volume is discretised and each grid node is considered to be a potential source position or a so-called image point.
    In other words, each image point represents a possible diffraction point from which seismic energy radiates. 
    The term "diffraction stacking" dates back to the works of Claerbout (1971, 1985) [3]_, [4]_ and Timoshin (1972) [7]_. 
    The concept was initially related to seismic migration and imaging of reflectors as a set of diffraction points (in the context of the exploding reflector principle). 
    Here, the diffraction stacking is related to imaging of real excitation sources.

    Diffraction stacking is applied to data with corrected event moveout (EMO),
    and can be done in several ways:
    
    1. Absolute-value based
    
    .. math::
        F(\mathbf{r},t) = \left| \sum_{R=1}^{N_R} A_R^{EMO}(t,\mathbf{r}) \right|,

    where :math:`\mathbf{r}` is a vector that defines a spatial position :math:`(x, y, z)` of the image point, 
    :math:`A_R^{EMO}(t,\mathbf{r})` represents the EMO-corrected data at the receiver :math:`R`, 
    and :math:`N_R` is a number of receivers [1]_.

    2. Squared-value based

    Similarly, but the stack is taken to the power of 2:   
    
    .. math::
        E(\mathbf{r},t) = \left[ \sum_{R=1}^{N_R} A_R^{EMO}(t,\mathbf{r}) \right]^2,
    
    3. Semblance-based

    Simple stacking can be further improved using a semblance-based approach.

    The semblance is a coherency or similarity measure and can be understood as the ratio of the total energy (the square of sum of amplitudes) 
    to the energy of individual traces (the sum of squares) [5]_:

    .. math::
        S(\mathbf{r},t) = \frac{\left[ \sum_{R=1}^{N_R} A_R^{EMO}(t,\mathbf{r}) \right]^2}
        {N_R \sum_{R=1}^{N_R} \left[ A_R^{EMO}(t,\mathbf{r}) \right]^2}    

    In order to suppress the effect of noise even better, it is possible to extend the semblance-based approach 
    by introducing a sliding time window :math:`W` over which the energy measures are summed:

    .. math::
        S_W(\mathbf{r},t,W) = \frac{\sum_{k=it-W}^{it+W}\left[ \sum_{R=1}^{N_R} A_R^{EMO}(t,\mathbf{r}) \right]^2}
        {N_R \sum_{k=it-W}^{it+W}\sum_{R=1}^{N_R} \left[ A_R^{EMO}(t,\mathbf{r}) \right]^2}

    where :math:`k` an index of the time-discretised signal within a sliding time interval 
    consisting of the :math:`2W + 1` samples, and :math:`it` is the index of time :math:`t` [8]_.

    Depending on the desired output, the function can output either a 3D volume and a location, 
    or a full 4D array (time dimension + 3 spatial dimensions).
    
    By default, a 3D image volume obtained as a maximum of the 4D image function, e.g.:

    .. math::
        F_s(\mathbf{r}) = \max_t F(\mathbf{r},t).
    
    Optionally, one can output a 3D image volume obtained as a mean of the 4D image function over time, e.g.:

    .. math::
        F_s(\mathbf{r}) = \underset{t}{\text{mean}}~F(\mathbf{r}, t).

    Another option is to output a 3D image volume obtained as a sum of squared values of the 4D image function over time, e.g.:

    .. math::
        F_s(\mathbf{r}) = \sum_t F^2(\mathbf{r},t).

    The full 4D output is useful for subsequent detection of multiple events.

    Polarity correction, if requested, is done using moment tensor inversion [2]_:

    .. math::
        F_{pc}(\mathbf{r},t) = \left|\sum_{R=1}^{N_R} \phi_R(\mathbf{r}) A_R^{EMO+PC} \right|,

    where 
    
    .. math::
        A_R^{EMO+PC}(t,\mathbf{r}) = \phi_R(\mathbf{r}) A_R \left(t + T_R(\mathbf{r})\right),

    and :math:`\phi_R(\mathbf{r})` are the defined through the :math:`\mathrm{sign}` function:

    .. math::
        \phi_R(\mathbf{r})=\mathrm{sign}(\mathbf{M}(\mathbf{r})\mathbf{G}_R^{\phantom{T}}(\mathbf{r})),\quad R=1\dots N_R,

    where :math:`\mathbf{M}(\mathbf{r})` is the vectorized moment tensor derived [6]_ as

    .. math::
        \mathbf{M}(\mathbf{r}) = \left(\sum_{R=1}^{N_R} \mathbf{G}_R^T(\mathbf{r})\mathbf{G}_R^{\phantom{T}}(\mathbf{r})\right)^{-1} \mathbf{d}(\mathbf{r}),

    where :math:`\mathbf{G}_R^{\phantom{T}}(\mathbf{r})` is the vectorized derivative of Green's function for receiver :math:`R` 
    and :math:`\mathbf{d}(\mathbf{r})` is a vector obtained from amplitudes of the corresponding waveforms:

    .. math::
        \mathbf{d}(\mathbf{r}) = \sum_{R=1}^{N_R} A_R(t+T_R(\mathbf{r})) \mathbf{G}_R^{\phantom{T}}(\mathbf{r}).

    More details on the described inversion including stability issues can be found in Zhebel & Eisner (2014) [9]_.

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

    .. [3] Claerbout, J. F. (1971). Toward a unified theory of reflector mapping. 
       Geophysics, 36(3), 467-481. https://doi.org/10.1190/1.1440185

    .. [4] Claerbout, J. (1985). Imaging the earth's interior. Oxford, England: Blackwell 
       Scientific Publications. https://sepwww.stanford.edu/sep/prof/iei2/
    
    .. [5] Neidell, N. S., & Taner, M. T. (1971). SEMBLANCE AND OTHER COHERENCY MEASURES 
       FOR MULTICHANNEL DATA. Geophysics, 36(3), 482–497. 
       https://doi.org/10.1190/1.1440186

    .. [6] Sipkin, S. A. (1982). Estimation of earthquake source parameters by the inversion of waveform data: synthetic waveforms. 
        Physics of the Earth and Planetary Interiors, 30(2–3), 242–259. https://doi.org/10.1016/0031-9201(82)90111-x

    .. [7] Timoshin Yu. V. (1972). Fundamentals of diffraction conversion of seismic 
       recordings. Moscow: Nedra. [In Russian].

    .. [8] Trojanowski, J., & Eisner, L. (2016). Comparison of migration‐based location 
       and detection methods for microseismic events. Geophysical Prospecting, 65(1), 
       47–63. https://doi.org/10.1111/1365-2478.12366

    .. [9] Zhebel, O., & Eisner, L. (2014). Simultaneous microseismic event localization and source mechanism determination. 
       Geophysics, 80(1), KS1–KS9. https://doi.org/10.1190/geo2014-0055.1

    """
    # Get sizes
    nx, ny, nz = n_xyz
    ngrid = nx * ny * nz
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
    if output_type not in ["max","mean","sumsq","full"]:
        raise ValueError(f"Output type is unknown: {output_type}")

    # Check stacking type
    if stack_type is None:
        stack_type = "absolute"
    if stack_type not in ["absolute","squared","semblance"]:
        raise ValueError(f"Diffraction stacking type is unknown: {stack_type}")

    # Check polarity correction
    if polcor_type is not None:
        if polcor_type not in ["mti"]:
            raise ValueError(f"Polarity correction type is unknown: {polcor_type}")
        if recs is None:
            raise ValueError(f"Receiver coordinates are required for polarity correction type {polcor_type}")

    # Reshape tt array
    ttg = tt.reshape(nr, -1)

    # Initialise image arrays
    ds_full = np.zeros((nt,ngrid))
    ds_im = np.zeros(ngrid)

    # Find time sample shifts for all grid points
    itshifts = np.round((ttg - ttg.min(axis=0))/dt)

    # Precompute vectorized Green tensor derivatives for grid points
    if polcor_type is not None:
        # Check sizes of recs
        if recs.shape[0] != 3:
            raise ValueError(f"Size of the receiver coordinates array is not consistent: {recs.shape}")
        if nr != recs.shape[1]:
            raise ValueError(f"Number of traces in data is not consistent with array of receiver coordinates: {nr} != {recs.shape[1]}")
        vgtd_grid = vgtd(x=x,y=y,z=z,recs=recs)
        gtg_inv_grid = mgtdinv(g=vgtd_grid)
            
    # Loop over grid points
    for igrid in range(ngrid):        

        # Perform event moveout correction for data
        data_mc = moveout_correction(data=data,
                                     itshifts=itshifts[:,igrid])

        # Perform polarity correction for data if needed
        if polcor_type is not None:
            data_pc = polarity_correction(data=data_mc,
                                          polcor_type=polcor_type,
                                          g=vgtd_grid[:,:,igrid],
                                          gtg_inv=gtg_inv_grid[:,:,igrid]
                                          )
        else:
            data_pc = data_mc

        # Perform stacking based on the type
        if stack_type == "absolute":
            ds = np.abs(np.sum(data_pc,axis=0))
        elif stack_type == "squared":       
            ds = (np.sum(data_pc,axis=0))**2            
        elif stack_type == "semblance":            
            ds = semblance_stack(data=data_pc,
                                 swsize=swsize)

        # Fill based on the output type
        if output_type == "max":
            ds_im[igrid] = np.max(ds)
        elif output_type == "mean":
            ds_im[igrid] = np.mean(ds)
        elif output_type == "sumsq":
            ds_im[igrid] = np.sum(ds**2)
        elif output_type == "full":                        
            ds_full[:, igrid] = ds            

    # Construct output based on its type
    if output_type in ["full"]:
        # Return full array and location as None
        hc = None
        return ds_full, hc
    else:
        ds_im_vol = ds_im.reshape(nx, ny, nz)            
        hc, _ = get_max_locs(ds_im_vol, n_max=nforhc, rem_edge=False)
        # Return 3D array and location
        return ds_im_vol, hc
