import pylops

from scipy.sparse.linalg import lsqr
from pylops.optimization.sparsity import *

from fracspy.location.utils import get_max_locs


def lsi(data, n_xyz, Op, niter=100, nforhc=10, verbose=True):
    """Least-squares imaging for microseismic source location

    This routine performs imaging of microseismic data by
    least-squares inversion using a Kirchhoff modelling operator.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of shape :math`n_r \times n_t`
    n_xyz : :obj:`tuple`
        Number of grid points in X-, Y-, and Z-axes for the imaging area
    Op : :obj:`pyfrac.modelling.kirchhoff.Kirchhoff`
        Kirchhoff operator
    niter : :obj:`int`, optional
        Number of iterations for inversion
    nforhc : :obj:`int`, optional
        Number of points for hypocenter
    verbose : :obj:`bool`, optional
        Verbosity (if ``True``, show iteration progression of the
        :func:`scipy.sparse.linalg.lsqr` solver)

    Returns
    -------
    inv : :obj:`numpy.ndarray`
        Inverted volume
    hc : :obj:`numpy.ndarray`
        Estimated hypocentral location

    """
    nx, ny, nz = n_xyz
    inv = (lsqr(Op, data.ravel(), iter_lim=niter, show=verbose)[0]).reshape(nx, ny, nz)
    hc, _ = get_max_locs(inv, n_max=nforhc, rem_edge=False)
    return inv, hc


def sparselsi(data, n_xyz, Op, niter=100, l1eps=1e2, nforhc=10, verbose=True):
    """Sparsity-promoting imaging for microseismic source location

    This routine performs imaging of microseismic data by
    sparsity-promoting inversion using a Kirchhoff modelling operator.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of shape :math`n_r \times n_t`
    n_xyz : :obj:`tuple`
        Number of grid points in X-, Y-, and Z-axes for the imaging area
    Op : :obj:`pyfrac.modelling.kirchhoff.Kirchhoff`
        Kirchhoff operator
    niter : :obj:`int`, optional
        Number of iterations for inversion
    l1eps : :obj:`float`, optional
        Weight of the L1 regularization term
    nforhc : :obj:`int`, optional
        Number of points for hypocenter
    verbose : :obj:`bool`, optional
        Verbosity (if ``True``, show iteration progression of the
        :func:`scipy.sparse.linalg.lsqr` solver)

    Returns
    -------
    inv : :obj:`numpy.ndarray`
        Inverted volume
    hc : :obj:`numpy.ndarray`
        Estimated hypocentral location

    """
    nx, ny, nz = n_xyz
    with pylops.disabled_ndarray_multiplication():
        inv = fista(Op, data.flatten(), niter=niter, eps=l1eps, show=verbose)[0].reshape(nx, ny, nz)
    hc, _ = get_max_locs(inv, n_max=nforhc, rem_edge=False)
    return inv, hc
