import numpy as np


def _get_centroid(array_xyz):
    return np.mean(array_xyz, axis=1)


def get_max_locs(ssimage, n_max=50, rem_edge=True, edgebuf=10):

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
