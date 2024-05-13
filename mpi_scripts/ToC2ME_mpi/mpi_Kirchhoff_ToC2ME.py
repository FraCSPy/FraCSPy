r"""
Kirchhoff microseismic location of TOC2ME dataset distributed over receivers.

Run as: export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; export NUMBA_NUM_THREADS=1; mpiexec -n 8 python mpi_Kirchhoff_ToC2ME.py
"""

import os
import numpy as np
import scipy as sp
import pandas as pd
import pylops_mpi
import time
import segyio
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
from mpi4py import MPI

from scipy.interpolate import RegularGridInterpolator
from pylops.utils.wavelets import *
from pylops_mpi.DistributedArray import local_split, Partition
from pyfrac.modelling.kirchhoff import Kirchhoff
from pyfrac.visualisation.plotting_support import explode_volume
from pyfrac.visualisation.traceviz import traceimage
from pyfrac.visualisation.eventimages import locimage3d

plt.close("all")


def run():
    comm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank() # worker id
    size = MPI.COMM_WORLD.Get_size() # number of worker

    tic = time.perf_counter()  

    if rank == 0:
        print(f'Distributed  microseismic location of TOC2ME data ({size} ranks)')

    # Create folder to save figures
    figdir = f'Figs'
    if rank == 0:
        if not os.path.exists(figdir):
            os.makedirs(figdir)

    # Receiver geometry
    rec_file = '/home/birniece/Documents/data/ToC2ME/Receivers/ToC2ME_Demo_recloc.txt'
    datum = -980

    recDF = pd.read_csv(rec_file)

    # determine lateral model extents from receiver geometry
    xmin = min(recDF['NAD83_X_m'])
    xmax = max(recDF['NAD83_X_m'])

    ymin = min(recDF['NAD83_Y_m'])
    ymax = max(recDF['NAD83_Y_m'])

    zmin = min(recDF['z_m'])
    zmax = max(recDF['z_m'])

    if rank == 0:
        print('X', xmin, xmax)
        print('Y', ymin, ymax)
        print('Z', zmin, zmax)

    rec_x = recDF['NAD83_X_m'] - xmin
    rec_y = recDF['NAD83_Y_m'] - ymin
    rec_z = abs(recDF['z_m'] + datum)

    recs = np.vstack((rec_x, rec_y, rec_z))
    recs = np.round(recs, decimals=-1)
    # recs = recs[:, :8]
    nr = recs.shape[1]

    if rank == 0:
        plt.figure()
        plt.scatter(recs[0], recs[1])
        plt.savefig(os.path.join(figdir, 'Receivers.png'), dpi=300)

    # Velocity Model
    vmod_file = '/home/birniece/Documents/data/ToC2ME/VelocityModels/ToC2MEVelModel.mat'
    mod_zprofile = sp.io.loadmat(vmod_file)

    z = mod_zprofile['z'] + datum
    vp_1D = mod_zprofile['vp']

    d_xyz = 15  # meter sampling
    mod_xmin = np.round(xmin, decimals=-1)
    mod_xmax = np.round(xmax, decimals=-1)

    mod_ymin = np.round(ymin, decimals=-1)
    mod_ymax = np.round(ymax, decimals=-1)

    mod_zmin = datum
    mod_zmax = 3500  # based on catalogue

    mod_x = np.arange(mod_xmin, mod_xmax, d_xyz)
    mod_y = np.arange(mod_ymin, mod_ymax, d_xyz)
    mod_z = np.arange(mod_zmin, mod_zmax, d_xyz)

    nx_mod = len(mod_x)
    ny_mod = len(mod_y)

    f_vp = sp.interpolate.interp1d(z.flatten(), vp_1D.flatten(), kind='previous')
    vp_log = f_vp(mod_z)

    vp_pancake = np.expand_dims(vp_log, axis=[1, 2])
    vp_pancake = vp_pancake.repeat(nx_mod, axis=2).repeat(ny_mod, axis=1).transpose([2, 1, 0])

    if rank == 0:
        plt.figure()
        plt.plot(vp_pancake[0, 0])
        plt.savefig(os.path.join(figdir, 'VP1D.png'), dpi=300)

    if rank == 0:
        plt.close('all')

    # Time axis and wavelet
    nt = 1001
    dt = 0.002
    t = np.arange(nt) * dt
    wav, wavt, wavc = ricker(t[:41], f0=20)

    # Choose how to split receivers to ranks
    nr_rank = local_split((nr, ), comm, Partition.SCATTER, 0)
    nr_ranks = np.concatenate(comm.allgather(nr_rank))
    irin_rank = np.insert(np.cumsum(nr_ranks)[:-1], 0, 0)[rank]
    irend_rank = np.cumsum(nr_ranks)[rank]
    print(f'Rank: {rank}, nr: {nr_rank}, irin: {irin_rank}, irend: {irend_rank}')

    # Split receivers into ranks
    rrank = recs[:, irin_rank:irend_rank]

    comm.Barrier()

    # Compute traveltimes
    if rank == 0:
        print('Compute traveltimes...', flush=True)

    nx, ny, nz = vp_pancake.shape
    dx, dy, dz = d_xyz, d_xyz, d_xyz
    x, y, z = np.arange(nx) * dx, np.arange(ny) * dy, np.arange(nz) * dz
    trav = Kirchhoff._traveltime_table(x=x,
                                       y=y,
                                       z=z,
                                       recs=rrank,
                                       vel=vp_pancake,
                                       mode='eikonal')

    if rank == 0:
        explode_volume(trav[:, 0].reshape(nx, ny, nz).transpose(2, 1, 0), cmap='tab10', t=20,
                       figsize=(15, 5))
        plt.savefig(os.path.join(figdir, 'Trav_rec.png'))

    # Operator
    KOp = Kirchhoff(z=z,
                    x=x,
                    y=y,
                    t=t,
                    srcs=rrank[:, :1],
                    recs=rrank,
                    vel=vp_pancake,
                    wav=wav,
                    wavcenter=wavc,
                    mode='byot',
                    trav=trav,
                    engine='numba')
    KOptot = pylops_mpi.MPIVStack(ops=[KOp, ])

    # Model data
    if rank == 0:
        print('Model data...', flush=True)

    sx, sy, sz = [ny // 4, nx // 2, nz // 2]
    microseismic = np.zeros((nx, ny, nz))
    microseismic[sx, sy, sz] = 1.

    microseismicdist = pylops_mpi.DistributedArray(global_shape=ny * nx * nz,
                                                   partition=pylops_mpi.Partition.BROADCAST)
    microseismicdist[:] = microseismic.flatten()

    frwddatadist = KOptot @ microseismicdist
    frwddata = frwddatadist.asarray().reshape(nr, nt)

    if rank == 0:
        ax = traceimage(frwddata, climQ=99.99)
        ax.set_title('Point Receivers')
        plt.savefig(os.path.join(figdir, 'Data.png'))

    # Adjoint
    if rank == 0:
        print('Compute adjoint...', flush=True)

    microseismicadjdist = KOptot.H @ frwddatadist
    microseismicadj = microseismicadjdist.asarray().reshape((nx, ny, nz))

    if rank == 0:
        fig, axs = locimage3d(microseismicadj, sx, sy, sz)
        plt.savefig(os.path.join(figdir, 'Adjoint.png'))

    # Inversion
    if rank == 0:
        print('Compute inverse...', flush=True)
    # Initializing x0 to zeroes
    x0 = pylops_mpi.DistributedArray(KOptot.shape[1],
                                     partition=pylops_mpi.Partition.BROADCAST)
    x0[:] = 0
    minv_dist = pylops_mpi.cgls(KOptot,
                                frwddatadist,
                                x0=x0,
                                niter=100,
                                show=True)[0]
    minv = minv_dist.asarray().reshape((nx, ny, nz))
    if rank == 0:
        fig, axs = locimage3d(minv, sx, sy, sz)
        plt.savefig(os.path.join(figdir, 'Inverse.png'))

    # d_inv_dist = KOptot @ minv_dist
    # d_inv = d_inv_dist.asarray().reshape(nstot, nr, nt)


if __name__ == '__main__':
    run()