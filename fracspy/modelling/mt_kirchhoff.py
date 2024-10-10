__all__ = [
    "MTSKirchhoff",
    "MTMKirchhoff",
]

from pylops.basicoperators import HStack, VStack
from pylops.utils import dottest

from fracspy.modelling.trueamp_kirchhoff import TAKirchhoff
from fracspy.mtinversion.greensfunction import *


def MTSKirchhoff(
        x,
        y,
        z,
        recs,
        t,
        wav,
        wavc,
        tt_table,
        Gz,
        Ms_scaling=1e6,
        engine='numba',
        checkdottest=True,
        ):
    r"""Moment Tensor Single Component Kirchhoff operator.

    Parameters
    ----------
    z : :obj:`numpy.ndarray`
        Depth axis
    x : :obj:`numpy.ndarray`
        Spatial axis
    y : :obj:`numpy.ndarray`
        Spatial axis
    recs : :obj:`numpy.ndarray`
        Receivers in array of size :math:`\lbrack 2 (3) \times n_r \rbrack`
        The first axis should be ordered as (``y``,) ``x``, ``z``.
    t : :obj:`numpy.ndarray`
        Time axis for data
    wav : :obj:`numpy.ndarray`
        Wavelet.
    wavcenter : :obj:`int`
        Index of wavelet center
    tt_table : :obj:`numpy.ndarray`
        Travel time table.
    G_z : :obj:`numpy.ndarray`
        Greens functions for a single component, i.e., z-component 
        For more information check _fracspy.mtinversion.greensfunctions.py_
    Ms_scaling : :obj:`float`
        Scaling to be incorporated in the MTI
    engine : :obj:`str`, optional
        Engine used for computations (``numpy`` or ``numba``).
    checkdottest : :obj:`Bool`, optional
        Run dot test to check operator 

    Returns
    -------
    Mstack_Op : Pylops operator
        Moment Tensor Operator
    """

    nr = recs.shape[1]
    Ms_Op = [TAKirchhoff(z=z,
                         x=x,
                         y=y,
                         t=t,
                         recs=recs,
                         wav=wav,
                         wavcenter=wavc,
                         wavfilter=True,
                         trav=tt_table.reshape(nr, -1).T,
                         amp=Ms_scaling * Gz[icomp].reshape(nr, -1).T.astype(np.float32),
                         engine=engine)
             for icomp in range(6)]
    # Combine to a single operator
    Mstack_Op = HStack(Ms_Op, forceflat=True)
    # check operator with dottest
    if checkdottest:
        _ = dottest(Mstack_Op, verb=True, atol=1e-5)
    return Mstack_Op


def MTMKirchhoff(
        x,
        y,
        z,
        recs,
        t,
        wav,
        wavc,
        tt_table,
        Gx,
        Gy,
        Gz,
        Ms_scaling = 1e6,
        engine='numba',
        checkdottest=True,
    ):
    r"""Moment Tensor Multi Component Kirchhoff operator.

    Parameters
    ----------
    z : :obj:`numpy.ndarray`
        Depth axis
    x : :obj:`numpy.ndarray`
        Spatial axis
    y : :obj:`numpy.ndarray`
        Spatial axis
    recs : :obj:`numpy.ndarray`
        Receivers in array of size :math:`\lbrack 2 (3) \times n_r \rbrack`
        The first axis should be ordered as (``y``,) ``x``, ``z``.
    t : :obj:`numpy.ndarray`
        Time axis for data
    wav : :obj:`numpy.ndarray`
        Wavelet.
    wavcenter : :obj:`int`
        Index of wavelet center
    tt_table : :obj:`numpy.ndarray`
        Travel time table.
    G_x : :obj:`numpy.ndarray`
        Greens functions for x-component
    G_y : :obj:`numpy.ndarray`
        Greens functions for y-component
    G_z : :obj:`numpy.ndarray`
        Greens functions for z-component
    Ms_scaling : :obj:`float`
        Scaling to be incorporated in the MTI
    engine : :obj:`str`, optional
        Engine used for computations (``numpy`` or ``numba``).
    checkdottest : :obj:`Bool`, optional
        Run dot test to check operator 

    Returns
    -------
    Mstack_Op : Pylops operator
        3C Moment Tensor Operator 
    """
        
    nr = recs.shape[1]

    # Build 6 True Amp Kirchoff Operators
    Mx_Op = [TAKirchhoff(z=z,
                         x=x,
                         y=y,
                         t=t,
                         recs=recs,
                         wav=wav,
                         wavcenter=wavc,
                         wavfilter=True,
                         trav=tt_table.reshape(nr, -1).T,
                         amp=Ms_scaling * Gx[icomp].reshape(nr, -1).T.astype(np.float32),
                         engine=engine)
             for icomp in range(6)]

    My_Op = [TAKirchhoff(z=z,
                         x=x,
                         y=y,
                         t=t,
                         recs=recs,
                         wav=wav,
                         wavcenter=wavc,
                         wavfilter=True,
                         trav=tt_table.reshape(nr, -1).T,
                         amp=Ms_scaling * Gy[icomp].reshape(nr, -1).T.astype(np.float32),
                         engine=engine)
             for icomp in range(6)]

    Mz_Op = [TAKirchhoff(z=z,
                         x=x,
                         y=y,
                         t=t,
                         recs=recs,
                         wav=wav,
                         wavcenter=wavc,
                         wavfilter=True,
                         trav=tt_table.reshape(nr, -1).T,
                         amp=Ms_scaling * Gz[icomp].reshape(nr, -1).T.astype(np.float32),
                         engine=engine)
             for icomp in range(6)]

    Mxstack_Op = HStack(Mx_Op, forceflat=True)
    Mystack_Op = HStack(My_Op, forceflat=True)
    Mzstack_Op = HStack(Mz_Op, forceflat=True)

    Mstack_Op = VStack([Mxstack_Op, Mystack_Op, Mzstack_Op])

    # check operator with dottest\
    if checkdottest:
        _ = dottest(Mstack_Op, verb=True, atol=1e-5)

    return Mstack_Op
