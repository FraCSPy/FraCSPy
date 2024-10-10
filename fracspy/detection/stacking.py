import numpy as np
#from fracspy.location.utils import *
from fracspy.location import Location

def maxdiffstack(data: np.ndarray,
                 x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray,
                 tt: np.ndarray, 
                 dt: float,                                                    
                 stack_type: str = None,
                 swsize: int = 0,
                 polcor_type: str = None,
                 recs: np.ndarray=None):
    r"""Maximum diffraction stack function for microseismic event detection.

    This routine uses diffraction stacking to obtain the time-dependent maximum stack function used for microseismic event detection.
    
    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of shape :math:`n_r \times n_t`    
    x : :obj:`numpy.ndarray`
        Imaging area grid vector in X-axis
    y : :obj:`numpy.ndarray`
        Imaging area grid vector in Y-axis
    z : :obj:`numpy.ndarray`
        Imaging area grid vector in Z-axis
    tt : :obj:`numpy.ndarray`
        Traveltime table of size :math:`n_r \times n_x \times n_y \times n_z`
    stack_type : :obj:`str`, optional, default: None
        Diffraction stacking type (imaging condition), default None is the same as ``squared``.
        Types: ``absolute`` (absolute value), ``squared`` (squared value), ``semblance`` (semblance-based).
    swsize : :obj:`int`, optional, default: 0
        Sliding time window size for semblance-based type, amount of time steps
    polcor_type : :obj:`str`, optional, default: None
        Polarity correction type to be used for amplitudes.
        None is default for no polarity correction, ``mti`` is for polarity correction based on moment tensor inversion.
    recs : :obj:`numpy.ndarray`, optional, default: None
        Array of shape (3, nrec) containing receiver coordinates.
        Must be provided if polcor_type is not None
    
    Returns
    -------
    msf : :obj:`numpy.ndarray`
        Time-dependent maximum stack function
    ds_full : :obj:`numpy.ndarray`
        Diffraction stack array reshaped to 4D (nt,nx,ny,nz)

    Notes
    -----
    For every time step :math:`t` the maximum of the diffraction stacking imaging function :math:`F(\mathbf{r},t)` over all potential locations
    :math:`\mathbf{r}` is evaluated [1]_:

    .. math::
        F_t(t) = \max_{\mathbf{r}} F(\mathbf{r},t),

    where :math:`F_t(t)` is referred to as the maximum stack function or MSF [2]_.

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

    """
    # Get sizes
    nx, ny, nz = len(x), len(y), len(z)
    ngrid = nx * ny * nz
    nr, nt = data.shape
    
    # Set up the location class
    L = Location(x, y, z)

    # Apply diffraction stacking to get full 4D function
    ds_full, _ = L.apply(data=data,
                         kind="diffstack",
                         x=x, y=y, z=z,
                         tt=tt, dt=dt,
                         stack_type=stack_type,
                         swsize=swsize,
                         polcor_type=polcor_type,
                         recs=recs,
                         output_type = "full")
    
    # Get the time-dependent MSF by taking maximum over the spatial dimension
    msf = np.max(ds_full, axis=1)

    # Return MSF and diffraction stack 4D array, i.e. reshaped to (nt,nx,ny,nz)         
    return msf, ds_full.reshape(nt, nx, ny, nz)

def stalta(tdf:np.ndarray,
           dt:float,
           stw:float,
           ltw:float,
           gtw:float=0):
    r"""STA/LTA computation.
    
    Computes Short Term Average / Long Term Average (STA/LTA) 
    for a time-dependent function provided the short time window (STW), long time window (LTW) and a gap time window (GTW) between them.

    Parameters
    ----------
    tdf : :obj:`numpy.ndarray`
        Time-dependent function of shape (nt,)
    dt : :obj:`float`
        Time step of the tdf
    stw : :obj:`float`
        Short time window in seconds (must be more than two time steps)
    ltw : :obj:`float`
        Long time window in seconds (must be more than two time steps)
    gtw : :obj:`float`, optional, default: 0
        Gap time window in seconds (must be non-negative)

    Returns
    -------
    slf : :obj:`numpy.ndarray`
        Time-dependent STA/LTA function
    sta : :obj:`numpy.ndarray`
        Time-dependent STA function
    lta : :obj:`numpy.ndarray`
        Time-dependent LTA function

    Notes
    -----
    The LTA and STA are defined as root mean squares of the signal in the LTW and STW with subtracted constant level.
    This constant level is the minimum between the average of the signal in the STW and LTW.
    The LTW goes first, then there is a GTW (if non-zero) and then goes the STW.
    The value of LTA/STA is calculated for the time moment on the border of the GTW and STW.
    The time function is padded appropriately on the both sides to have 
    a full-length preceding LTW + GTW and a full-length following STW.

    """
    # Get size of the function
    nt = len(tdf)

    # Check if nt is at least 4 time steps
    if nt < 3 * dt:
        raise ValueError("Input time-dependent function must be more than three time steps.")

    # Check if stw and ltw are more than two time steps
    if stw < 2 * dt or ltw < 2 * dt:
        raise ValueError("STW and LTW must be more than two time steps.")

    # Check if gtw is non-negative
    if gtw < 0:
        raise ValueError("GTW must be non-negative.")

    # Get sizes of stw, ltw and gtw, rounding and converting to integers
    nstw = int(np.around(stw / dt))
    nltw = int(np.around(ltw / dt))
    ngtw = int(np.around(gtw / dt))

    # Initialise STA/LTA function (SLF)
    slf = np.zeros_like(tdf)

    # Initialise arrays for STA/LTA, STA, LTA
    slf = np.zeros(nt)
    sta = np.zeros(nt)
    lta = np.zeros(nt)

    # Find low level of amplitudes to substitute for zeros in tdf
    lowlevel = max(np.finfo(float).eps, 1e-3 * np.median(np.abs(tdf)))

    # Substitute zeros with low level in tdf to avoid nans
    tdf[tdf == 0] = lowlevel
    
    # Avoid boundary problems by padding the input signal appropriately
    padded_tdf = np.pad(tdf, (nltw + ngtw, nstw), mode='reflect')
    
    # Compute LTA and STA as the root mean square (RMS) in the respective windows
    for it in range(nt):
        # Indices for the LTA window
        lta_start = it
        lta_end = it + nltw
        
        # Indices for the STA window
        sta_start = it + nltw + ngtw
        sta_end = sta_start + nstw
        
        # Calculate RMS for LTA and STA
        level = min(np.mean(padded_tdf[lta_start:lta_end]),np.mean(padded_tdf[sta_start:sta_end]))
        lta[it] = np.sqrt(np.mean((padded_tdf[lta_start:lta_end]-level) ** 2))
        sta[it] = np.sqrt(np.mean((padded_tdf[sta_start:sta_end]-level) ** 2))
        
        # Calculate STA/LTA ratio, avoid division by zero        
        slf[it] = sta[it] / lta[it] if lta[it] != 0 else 0
    
    return slf, sta, lta

def detect_peaks(msf:np.ndarray,
                 slf:np.ndarray,
                 slt:float):
    r"""Detect peaks in the maximum stack function based on STA/LTA function.
    
    Detect local maxima in the time-dependent maximum stack function (MSF) based on the analysis of STA/LTA function (SLF) and an STA/LTA thereshold (SLT).

    Parameters
    ----------
    msf : :obj:`numpy.ndarray`
        Time-dependent maximum stack function of shape (nt,)
    slf : :obj:`numpy.ndarray`
        Time-dependent STA/LTA function of shape (nt,)    
    slt : :obj:`float`
        STA/LTA threshold, must be non-negative
    
    Returns
    -------
    idp : :obj:`numpy.ndarray`
        Integer array of time indices of determined peaks
    
    Raises
    ------
    ValueError :
        if slt is negative

    Notes
    -----
    The SLF is compared to the SLT. 
    If all values of the SLF are below the SLT, the array of time indices of determined peaks (`idp`) is set to be empty.
    Otherwise, there will be one or more regions where SLf exceeds SLT. 
    For each of these regions, sequentially, the algorithm finds an index of the maximum value of the MSF and saves it the `idp`.
    """
    # Check SLT
    if slt<0:
        raise ValueError(f"STA/LTA threshold must be non-negative, but it is {slt}")

    # Identify regions where SLF exceeds the threshold SLT
    above_threshold = slf > slt

    if not np.any(above_threshold):
        # Return empty array if no values in SLF exceed the threshold
        return np.array([], dtype=int)
    
    # Find the boundaries of regions where SLF exceeds the threshold
    regions = np.where(np.diff(above_threshold.astype(int)))[0] + 1

    # Include start and end boundaries if necessary
    if above_threshold[0]:
        regions = np.insert(regions, 0, 0)
    if above_threshold[-1]:
        regions = np.append(regions, len(above_threshold))

    # Initialize the array for storing peak indices
    idp = []

    # Iterate over the regions to detect peaks
    for i in range(0, len(regions), 2):
        start = regions[i]
        end = regions[i+1]
        # Find the index of the maximum value of MSF in this region
        max_index = np.argmax(msf[start:end]) + start
        idp.append(max_index)
    
    return np.array(idp, dtype=int)

def estimate_origin_times(idp: np.ndarray,
                          ds_full: np.ndarray,
                          tt: np.ndarray,
                          dt: float,
                          x: np.ndarray,
                          y: np.ndarray,
                          z: np.ndarray):
    r"""Estimate origin times of the events detected with diffraction stacking.
    
    Estimate origin times of the events using traveltimes and the time indices 
    determined from the maximum stack function (MSF) using STA/LTA analysis.

    Parameters
    ----------
    idp : :obj:`numpy.ndarray`
        Integer array of time indices of determined peaks
    ds_full : :obj:`numpy.ndarray`
        Diffraction stack full 4D array of shape (nt, nx, ny, nz)
    tt : :obj:`numpy.ndarray`
        Traveltime table of size :math:`n_r \times n_x \times n_y \times n_z`    
    dt : :obj:`float`
        Time step
    x : :obj:`numpy.ndarray`
        Imaging area grid vector in X-axis
    y : :obj:`numpy.ndarray`
        Imaging area grid vector in Y-axis
    z : :obj:`numpy.ndarray`
        Imaging area grid vector in Z-axis
    
    Returns
    -------
    eot : :obj:`numpy.ndarray`
        Array of estimated origin times for each of the determined peaks
        
    Notes
    -----
    For each determined MSF peak, the origin time :math:`t_{est}` of the corresponding event can be determined from peak time :math:`t_{peak}` as 

    .. math::
        \mathbf{t}_{est} = t_{peak} - \min_{R} T_R(\mathbf{r}_{peak}).

    where :math:`T_R(\mathbf{r}_{peak})` is the traveltime to receiver :math:`R` from :math:`\mathbf{r}_{peak}` - 
    the location determined from the maximum of the 4D imaging function :math:`F(\mathbf{r},t)` at the time moment :math:`t_{peak}`:

    .. math::
        \mathbf{r}_{peak} = \arg\!\max_{\mathbf{r}} F(\mathbf{r},t_{peak}).
    
    """
    # Get sizes
    nt, nx, ny, nz = ds_full.shape

    # Number of detected peaks
    nde = len(idp)

    # Create data time vector
    t = np.arange(0, nt * dt, dt)

    # Initialize the eot array
    eot = np.zeros(nde)

    # Loop over detections
    for ide in range(nde):
        # Get time index
        it = idp[ide]

        # Find spatial indices corresponding to the maximum of ds_full at time index it
        ix, iy, iz = np.unravel_index(np.argmax(ds_full[it, :, :, :]), ds_full[it, :, :, :].shape)

        # Get the estimated origin time
        eot[ide] = t[it] - np.min(np.squeeze(tt[:,ix,iy,iz]))

    return eot

def diffstack_detect(data: np.ndarray,
                     x: np.ndarray,
                     y: np.ndarray,
                     z: np.ndarray,
                     tt: np.ndarray, 
                     dt: float,       
                     stw: float,
                     ltw: float,
                     gtw: float=0,
                     slt: float=3,                                             
                     stack_type: str = None,
                     swsize: int = 0,
                     polcor_type: str = None,
                     recs: np.ndarray=None):
    r"""Microseismic event detection based on diffraction stacking.

    The time-dependent maximum stack function (MSF) obtained by diffraction stacking is analysed using STA/LTA.
    
    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of shape :math:`n_r \times n_t`    
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
    stw : :obj:`float`
        Short time window in seconds (must be more than two time steps)
    ltw : :obj:`float`
        Long time window in seconds (must be more than two time steps)
    gtw : :obj:`float`, optional, default: 0
        Gap time window in seconds (must be non-negative)
    gtw : :obj:`float`, optional, default: 0
        Gap time window in seconds (must be non-negative)
    stack_type : :obj:`str`, optional, default: None
        Diffraction stacking type (imaging condition), default None is the same as ``squared``.
        Types: ``absolute`` (absolute value), ``squared`` (squared-value), ``semblance`` (semblance-based).
    swsize : :obj:`int`, optional, default: 0
        Sliding time window size for semblance-based type, amount of time steps
    polcor_type : :obj:`str`, optional, default: None
        Polarity correction type to be used for amplitudes.
        None is default for no polarity correction, ``mti`` is for polarity correction based on moment tensor inversion.
    recs : :obj:`numpy.ndarray`, optional, default: None
        Array of shape (3, nrec) containing receiver coordinates.
        Must be provided if polcor_type is not None
    
    Returns
    -------
    msf : :obj:`numpy.ndarray`
        Time-dependent maximum stack function
    slf : :obj:`numpy.ndarray`
        Time-dependent STA/LTA function
    idp : :obj:`numpy.ndarray`
        Integer array of time indices of determined peaks
    eot : :obj:`numpy.ndarray`
        Array of estimated origin times for each of the determined peaks
    ds_full : :obj:`numpy.ndarray`
        Diffraction stack full 4D array of shape (nt, nx, ny, nz)

    Notes
    -----
    The local maxima of the maximum stack function [1]_ (MSF) occur at the times linked to the origin times of microseismic events. 
    These local maxima can be found by the STA/LTA (Short Term Average / Long Term Average) method [3]_.
    Local maxima are detected by measuring the ratio of average stack energy in short and long sliding time windows and
    comparing this ratio with the pre-defined STA/LTA threshold (SLT). 
    The number of detected events depends on the signal-to-noise ratio (SNR) of the maximum stack function measured by STA/LTA [2]_.
    If the SLT is too high, too few events are detected, but there is a high certainty that the detected events are real.
    If a too low threshold is used, false events will be detected as real.
    
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

    .. [3] Trnkoczy, A. (2012). Understanding and parameter setting of STA/LTA trigger algorithm. 
        In: Bormann, P. (Ed.), New Manual of Seismological Observatory Practice 2 (NMSOP-2), 
        Potsdam: Deutsches GeoForschungsZentrum GFZ, 1-20.
        https://doi.org/10.2312/GFZ.NMSOP-2_IS_8.1

    """
    # Get maximum stack function
    msf,ds_full = maxdiffstack(data=data,
                         x=x,
                         y=y,
                         z=z,
                         tt=tt,
                         dt=dt,                                                    
                         stack_type=stack_type)
    
    # Compute STA/LTA
    slf,_,_ = stalta(tdf=msf,
                     dt=dt,
                     stw=stw,
                     ltw=ltw,
                     gtw=gtw)

    # Detect msf peaks in triggered zones
    idp = detect_peaks(msf=msf,
                       slf=slf,
                       slt=slt)

    # Estimate event origin times
    eot = estimate_origin_times(idp=idp,
                                ds_full=ds_full,
                                tt=tt,
                                dt=dt,                                
                                x=x,
                                y=y,
                                z=z)

    return msf,slf,idp,eot,ds_full