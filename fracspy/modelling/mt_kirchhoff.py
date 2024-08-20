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
