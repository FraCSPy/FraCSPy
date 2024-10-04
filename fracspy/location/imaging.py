import numpy as np
import torch 
from scipy.sparse.linalg import lsqr
import pylops
from pylops.optimization.sparsity import *
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR
from fracspy.location.utils import get_max_locs


def lsi(data, n_xyz, Op, niter=100, nforhc=10, verbose=False):
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


def sparselsi(data, n_xyz, Op, niter=100, l1eps=1e2, nforhc=10, verbose=False):
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


def xcorr_of(x, y):
    """Pearson correlation coefficient

    Compute the Pearson correlation coefficient between two datasets

    Parameters
    ----------
    x : :obj:`torch.Tensor`
        First dataset
    y : :obj:`torch.Tensor`
        Second dataset

    Returns
    -------
    r2 : :obj:`float`
        Pearson correlation coefficient

    """
    x = x / torch.linalg.norm(x)
    y = y / torch.linalg.norm(y)
    r2 = - torch.sum(torch.mul(x, y))
    return r2


def xcorri(data, n_xyz, Op, niter=100, l1eps=8e-1,
           lr=1e-5, nforhc=10):
    """Cross-correlation-based imaging for microseismic source location

    This routine performs imaging of microseismic data by inversion using an
    objective function based on the Pearson correlation coefficient. This idea
    is borrowed from the field of seismic migration, and more specifically from
    [1]_, and is intended to create a location algorithm that is less sensitive to
    inaccuracies in the knowledge of the source signature.

    .. [1] Zhang, Y., and Duan, L., and  Xie, Y. "A stable and practical implementation
       of least-squares reverse time migration", Geophysics, Vol. 80, 1, pp. 1JF-Z39, 2015.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of shape :math`n_r \times n_t`
    n_xyz : :obj:`tuple`
        Number of grid points in X-, Y-, and Z-axes for the imaging area
    Op : :obj:` pyfrac.modelling.kirchhoff.Kirchhoff`
        Kirchhoff operator
    niter : :obj:`int`, optional
        Number of iterations
    l1eps : :obj:`float`, optional
        Weight of the L1 regularization term
    lr : :obj:`int`, optional
        Learning rate used by the optimizer
    nforhc : :obj:`int`, optional
        Number of points for hypocenter

    Returns
    -------
    mls_torch : :obj:`numpy.ndarray`
        Migrated volume
    hc : :obj:`numpy.ndarray`
        Estimated hypocentral location
    dls_torch : :obj:`numpy.ndarray`
        Predicted data volume
    losshist : :obj:`list`
        Loss history

    """

    # Initialise with migrated image
    migrated = (Op.H @ data).reshape(n_xyz)
    dmigrated = Op @ migrated.ravel()
    scaling = data.max() / dmigrated.max()
    m = torch.from_numpy(migrated.copy().ravel() * scaling)
    m.requires_grad = True  # make sure we compute the gradient with respect to m

    # Initialise torch operator and torch tensors
    TOp = pylops.TorchOperator(Op)
    dobs = torch.from_numpy(data.copy().ravel())

    # Optimization
    optimizer = torch.optim.SGD([m], lr=lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    losshist = np.zeros(niter)
    pbar = tqdm(range(niter))
    for i in pbar:
        optimizer.zero_grad()
        d = TOp(m)

        # data term
        lossd = xcorr_of(d, dobs)

        # L1 regularization
        reg = 0.
        if l1eps > 0:
            reg = torch.sum(torch.abs(m))

        # total loss
        loss = lossd + l1eps * reg

        loss.backward()
        optimizer.step()
        scheduler.step()
        losshist[i] = loss.item()

        pbar.set_postfix({"Data Loss": lossd.item(), "L1reg Loss": reg.item(), "Total Loss": loss.item()})

    dls_torch = d.detach().cpu().numpy().reshape(data.shape)
    mls_torch = m.detach().cpu().numpy().reshape(n_xyz)
    hc, hcs = get_max_locs(mls_torch, n_max=nforhc, rem_edge=False)

    return mls_torch, hc, dls_torch, losshist
