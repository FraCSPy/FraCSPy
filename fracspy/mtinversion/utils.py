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
    """Moment

    Parameters
    ----------
    mt_image_set
    location_indices

    Returns
    -------

    """
    mxx = mt_image_set[0][location_indices[0], location_indices[1], location_indices[2]]
    myy = mt_image_set[1][location_indices[0], location_indices[1], location_indices[2]]
    mzz = mt_image_set[2][location_indices[0], location_indices[1], location_indices[2]]
    mxy = mt_image_set[3][location_indices[0], location_indices[1], location_indices[2]]
    mxz = mt_image_set[4][location_indices[0], location_indices[1], location_indices[2]]
    myz = mt_image_set[5][location_indices[0], location_indices[1], location_indices[2]]

    return (mxx,myy,mzz,mxy,mxz,myz)


def expected_sloc_from_mtwi(mt_image_set, nforhc=5, rem_edge=True, edgebuf=1, absval=True):
    energy_images = np.sum(abs(mt_image_set), axis=0)
    hc, hcs = get_max_locs(energy_images,
                           n_max=nforhc,
                           rem_edge=rem_edge,
                           edgebuf=edgebuf,
                           absval=absval)
    return hc, hcs

# Determine seismic moment m0 and local magnitude mw
def get_magnitude(mt):
    mt_array = np.array(mt)  # Convert the list to a NumPy array
    m0 = np.sqrt(np.sum(mt_array ** 2))  # Use NumPy functions for better performance
    mw = ((2/3)*(np.log10(m0) - 9.1))
    return m0, mw