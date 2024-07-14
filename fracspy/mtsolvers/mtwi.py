import numpy as np
from scipy.sparse.linalg import lsqr
from pylops.basicoperators import HStack, VStack
from pylops.utils import dottest
from pylops.optimization.sparsity import *
from fracspy.modelling.trueamp_kirchhoff import TAKirchhoff


def singlecomp_pwave_mtioperator(x,
                                 y,
                                 z,
                                 recs,
                                 t,
                                 wav,
                                 wavc,
                                 tt_table,
                                 Gz,
                                 Ms_scaling = 1e6,
                                 engine='numba',
                                 checkdottest=True
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


def multicomp_pwave_mtioperator(x,
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
                          checkdottest=True
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


def frwrd_mtmodelling(mt_cube, Mstack_Op, nr, nt, multicomp=True):
    # Generated Data
    data = Mstack_Op @ mt_cube.ravel()
    if multicomp:
        data = data.reshape(3, nr, nt)
    else:
        data = data.reshape(nr, nt)
    return data

def adjoint_mtmodelling(data, Mstack_Op, nxyz):
    mt_adj = Mstack_Op.H @ data.ravel()
    mt_adj = mt_adj.reshape([6, nxyz[0], nxyz[1], nxyz[2]])
    return mt_adj


def lsqr_mtsolver(data, Mstack_Op, nxyz):
    mt_inv = lsqr(Mstack_Op, data.ravel(), iter_lim=50, atol=0, btol=0, show=True)[0]
    mt_inv = mt_inv.reshape([6, nxyz[0], nxyz[1], nxyz[2]])
    return mt_inv

def fista_mtsolver(data, Mstack_Op, nxyz, fista_niter=100, fista_damping=1e-13, verbose=True):
    mt_fista = fista(Mstack_Op,
                         data.ravel(),
                         np.zeros([6, nxyz[0], nxyz[1], nxyz[2]]).ravel(),  # initialise at zeroes
                         niter=fista_niter,
                         eps=fista_damping,
                         show=verbose)[0]
    mt_fista = mt_fista.reshape([6, nxyz[0], nxyz[1], nxyz[2]])
    return mt_fista
