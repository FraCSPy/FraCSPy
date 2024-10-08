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
              recs: np.ndarray=None,
              output_type: str = "ms"):
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
        Diffraction stacking type (imaging condition), default None is the same as "absolute" (absolute value).
        Types: "absolute" (absolute value), "squared" (squared value), "semblance" (semblance-based).
    swsize : :obj:`int`, optional, default: 0
        Sliding time window size for semblance-based type, amount of time steps
    polcor_type : :obj:`str`, optional, default: None
        Polarity correction type to be used for amplitudes.
        None is default for no polarity correction, "mti" is for polarity correction based on moment tensor inversion.
    recs : :obj:`numpy.ndarray`, optional, default: None
        Array of shape (3, nrec) containing receiver coordinates.
        Must be provided if polcor_type is not None
    output_type : :obj:`str`, optional, default: "ms"
        Defines the output type:
        - "ms" (output only the time-dependent maximum stack function `msf` (nt,),            
        - "full" (output the time-dependent maximum stack function `msf` and the full 4D imaging function: (nx,ny,nz,nt).

    Returns
    -------
    msf : :obj:`numpy.ndarray`
        Time-dependent maximum stack function    
    ds_full : :obj:`numpy.ndarray`
        Diffraction stack 4D array (nt,nx,ny,nz)

    Raises
    ------
   

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
    
    print(ds_full.shape)

    # Get the time-dependent MSF by taking maximum over spatial dimensions
 #   msf = np.max(ds_full, axis=1)
    
#    return msf, ds_full

    return ds_full