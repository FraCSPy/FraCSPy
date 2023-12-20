from pyfrac.locationsolvers.localisationutils import get_max_locs
from scipy.sparse.linalg import lsqr
import pylops
from pylops.optimization.sparsity  import *


def migration(Op, data, n_xyz, nforhc=10):
    ''' Perfect inverse < impossible in reality

    Parameters
    ----------
    Op - PyLops Kirchhoff operator
    data - seismic data [nr,nt]
    n_xyz - dimensions of subsurface model/migrated image
    nforhc - number of points for hypocenter

    Returns
    -------

    '''
    nx, ny, nz = n_xyz
    migrated = (Op.H @ data).reshape(nx, ny, nz)
    hc, _ = get_max_locs(migrated, n_max=nforhc, rem_edge=False)
    return migrated, hc


def lsqr_migration(Op, data, n_xyz, niter=100, nforhc=10, verbose=True):
    '''

    Parameters
    ----------
    Op
    data
    n_xyz
    niter
    nforhc
    verbose

    Returns
    -------

    '''
    nx, ny, nz = n_xyz
    inv = (lsqr(Op, data.ravel(), iter_lim=niter, show=verbose)[0]).reshape(nx, ny, nz)
    hc, _ = get_max_locs(inv, n_max=nforhc, rem_edge=False)
    return inv, hc


def fista_migration(Op, data, n_xyz, niter=100, fista_eps=1e2, nforhc=10, verbose=True):
    nx, ny, nz = n_xyz
    with pylops.disabled_ndarray_multiplication():
        fista_mig = fista(Op, data.flatten(), niter=niter, eps=fista_eps, show=verbose)[0].reshape(nx, ny, nz)
    hc, _ = get_max_locs(fista_mig, n_max=nforhc, rem_edge=False)
    return fista_mig, hc
