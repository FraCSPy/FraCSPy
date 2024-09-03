import numpy as np
from fracspy.location.utils import get_max_locs


# Dictionary of the MT components values
MT_comp_dict = [{'elementID': 0, 'laymans': 'xx', 'pq': [0, 0], 'ODscaler': 1, 'MCweighting': 1},
                {'elementID': 1, 'laymans': 'yy', 'pq': [1, 1], 'ODscaler': 1, 'MCweighting': 1},
                {'elementID': 2, 'laymans': 'zz', 'pq': [2, 2], 'ODscaler': 1, 'MCweighting': 1},
                {'elementID': 3, 'laymans': 'xy', 'pq': [0, 1], 'ODscaler': 2, 'MCweighting': 1},
                {'elementID': 4, 'laymans': 'xz', 'pq': [0, 2], 'ODscaler': 2, 'MCweighting': 1},
                {'elementID': 5, 'laymans': 'yz', 'pq': [1, 2], 'ODscaler': 2, 'MCweighting': 1},
                ]


def get_mt_at_loc(mt_image_set, location_indices):
    """Moment tensor at specific location

    Extract moment tensor at a given location from moment tensor component images

    Parameters
    ----------
    mt_image_set : :obj:`numpy.ndarray`
        Moment tensor component images of size :math:`n_c \times n_x \times n_y \times n_z`
    location_indices : :obj:`tuple`
        Indices of location to extract moment tensor from the moment tensor component images

    Returns
    -------
    M : :obj:`tuple`
        Extracted moment tensor

    """
    mxx = mt_image_set[0][location_indices[0], location_indices[1], location_indices[2]]
    myy = mt_image_set[1][location_indices[0], location_indices[1], location_indices[2]]
    mzz = mt_image_set[2][location_indices[0], location_indices[1], location_indices[2]]
    mxy = mt_image_set[3][location_indices[0], location_indices[1], location_indices[2]]
    mxz = mt_image_set[4][location_indices[0], location_indices[1], location_indices[2]]
    myz = mt_image_set[5][location_indices[0], location_indices[1], location_indices[2]]

    M = (mxx, myy, mzz, mxy, mxz, myz)
    return M


def get_mt_max_locs(mt_image_set, n_max=50, rem_edge=True, edgebuf=10):
    """Source location from moment tensor component images

    Compute the source location from the sum of the absolute values of the
    six moment tensor component images.

    Parameters
    ----------
    mt_image_set : :obj:`numpy.ndarray`
        Moment tensor component images of size :math:`n_c \times n_x \times n_y \times n_z`
    n_max : :obj:`int`, optional
        Number of maximum values to extract (if ``n_max>1``, the centroid of these values
        will be computed and provided as the estimated source location)
    rem_edge : :obj:`bool`, optional
        Remove edges of volume
    edgebuf : :obj:`int`, optional
        Number of grid points to remove from each edge if ``rem_edge=True``

    Returns
    -------
    ev_loc : :obj:`tuple`
        Most likely source location
    ev_locs : :obj:`tuple`
        `n_max` most likely source locations

    """
    energy_images = np.sum(np.abs(mt_image_set), axis=0)
    ev_loc, ev_locs = get_max_locs(energy_images,
                                   n_max=n_max,
                                   rem_edge=rem_edge,
                                   edgebuf=edgebuf,
                                   absval=False)
    return ev_loc, ev_locs


def get_magnitude(mt):
    """Seismic moment and local magnitude

    Determine seismic moment ``m0`` and local magnitude ``mw``
    from moment tensor array.

    Parameters
    ----------
    mt : :obj:`numpy.ndarray`
        Moment tensor

    Returns
    -------
    m0 : :obj:`tuple`
        Seismic moment
    mw : :obj:`tuple`
        Local magnitude

    """
    mt_array = np.array(mt)
    m0 = np.sqrt(np.sum(mt_array ** 2))
    mw = ((2 / 3) * (np.log10(m0) - 9.1))
    return m0, mw