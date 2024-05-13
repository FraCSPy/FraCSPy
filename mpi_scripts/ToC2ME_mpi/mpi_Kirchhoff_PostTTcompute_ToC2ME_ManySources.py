r"""
Kirchhoff microseismic location of TOC2ME dataset distributed over receivers.

Run as: export OMP_NUM_THREADS=2; export MKL_NUM_THREADS=2; export NUMBA_NUM_THREADS=2; mpiexec -n 4 python mpi_Kirchhoff_PostTTcompute_ToC2ME_ManySources.py
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

    # Define folder locations
    ttdir = f'TTdata'
    outdir = '/media/birniece/Extreme SSD/ToC2ME/Eikonal_Modelled_Data'

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
    if rank == 0: print(mod_xmax, mod_ymax, mod_zmax)

    nx_mod = len(mod_x)
    ny_mod = len(mod_y)

    f_vp = sp.interpolate.interp1d(z.flatten(), vp_1D.flatten(), kind='previous')
    vp_log = f_vp(mod_z)
    vp_pancake = np.expand_dims(vp_log, axis=[1, 2])
    vp_pancake = vp_pancake.repeat(nx_mod, axis=2).repeat(ny_mod, axis=1).transpose([2, 1, 0])

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

    # Load pre-computed traveltimes
    if rank == 0:
        print('Load pre-computed traveltimes...', flush=True)

    nx, ny, nz = vp_pancake.shape
    dx, dy, dz = d_xyz, d_xyz, d_xyz
    x, y, z = np.arange(nx) * dx, np.arange(ny) * dy, np.arange(nz) * dz
    trav = np.load(os.path.join(ttdir, 'TT_rank%i.npy'%rank))

    if rank == 0:
        print('Done pre-computed traveltimes...', flush=True)


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

    # RANDOMLY SELECT SOURCES
    nsrc_large = 8192
    d_xzy = [d_xyz,d_xyz,d_xyz]

    # LARGE SPACE
    sx_min = 3000 - mod_xmin
    sx_max = 6000 - mod_xmin
    sy_min = 4000 - mod_ymin
    sy_max = 8000 - mod_ymin
    sz_min = 2000 + datum
    sz_max = 3000 + datum

    if rank == 0: print(sx_min, sy_min, sz_min)
    if rank == 0: print(sx_max, sy_max, sz_max)
    grid_sx_locs = np.arange(sx_min, sx_max, d_xzy[0])
    grid_sy_locs = np.arange(sy_min, sy_max, d_xzy[2])
    grid_sz_locs = np.arange(sz_min, sz_max, d_xzy[1])

    sx, sy, sz = np.meshgrid(grid_sx_locs,
                             grid_sy_locs,
                             grid_sz_locs)
    srclocs_large = np.vstack((sx.flatten(), sy.flatten(), sz.flatten())).T  # y is y, I fix the SOFI ordering later
    # print(srclocs_large.shape)
    rng = np.random.default_rng(seed=20)
    rng.shuffle(srclocs_large, axis=0)
    srclocs_large_selected = srclocs_large[:nsrc_large]
    # print('rank: ', rank, ' srclocs_large_selected: ', srclocs_large_selected[:2])

    for src in srclocs_large_selected:
        sx, sy, sz = [int((src[0] - mod_xmin)/d_xzy[0]),
                      int((src[1] - mod_ymin)/d_xzy[1]),
                      int((src[2] - mod_zmin)/d_xzy[2]),
                      ]
        microseismic = np.zeros((nx, ny, nz))
        # print('rank: ', rank, ' Model: ', (nx, ny, nz), ' Micro: ', microseismic.shape, ' Source: ', (sx, sy, sz))
        microseismic[sx, sy, sz] = 1.

        microseismicdist = pylops_mpi.DistributedArray(global_shape=ny * nx * nz,
                                                       partition=pylops_mpi.Partition.BROADCAST)
        microseismicdist[:] = microseismic.flatten()

        ticf = time.perf_counter()
        frwddatadist = KOptot @ microseismicdist
        frwddata = frwddatadist.asarray().reshape(nr, nt)
        comm.Barrier()
        tocf = time.perf_counter()

        if rank == 0:
            # Save data
            np.savez(os.path.join(outdir, 'ToC2ME_Eik_%i_%i_%i.npz'%(src[0], src[1], src[2])),
                     loc=src,
                     data=frwddata
                     )

    toc = time.perf_counter()
    if rank == 0:
        print(f'Forward elapsed time: {tocf-ticf} s')
        print(f'Total elapsed time: {toc-tic} s')


if __name__ == '__main__':
    run()