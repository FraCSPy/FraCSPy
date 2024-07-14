from fracspy.locationsolvers.localisationutils import get_max_locs
from fracspy.locationsolvers.localisationutils import moveout_correction
from fracspy.locationsolvers.localisationutils import semblance_stack
from scipy.sparse.linalg import lsqr
import pylops
from pylops.optimization.sparsity  import *
import numpy as np

def migration(Op, data, n_xyz, nforhc=10):
    ''' Kirchhoff migration for microseismic source location

    Parameters
    ----------
    Op - PyLops Kirchhoff operator
    data - seismic data [nr,nt]
    n_xyz - dimensions of subsurface model/migrated image
    nforhc - number of points for hypocenter

    Returns
    -------
    migrated - Migrated volume
    hc - determined hypocentral location

    '''
    nx, ny, nz = n_xyz
    migrated = (Op.H @ data).reshape(nx, ny, nz)
    hc, _ = get_max_locs(migrated, n_max=nforhc, rem_edge=False)
    return migrated, hc


def lsqr_migration(Op, data, n_xyz, niter=100, nforhc=10, verbose=True):
    '''  LSQR Kirchhoff-based inversion for microseismic source location

    Parameters
    ----------
    Op - PyLops Kirchhoff operator
    data - seismic data [nr,nt]
    n_xyz - dimensions of subsurface model/migrated image
    niter - number of iterations for inversion, default 100
    nforhc - number of points for hypocenter, default 10
    verbose - run in verbose mode, default True

    Returns
    -------
    inv - inverted volume
    hc - determined hypocentral location

    '''
    nx, ny, nz = n_xyz
    inv = (lsqr(Op, data.ravel(), iter_lim=niter, show=verbose)[0]).reshape(nx, ny, nz)
    hc, _ = get_max_locs(inv, n_max=nforhc, rem_edge=False)
    return inv, hc


def fista_migration(Op, data, n_xyz, niter=100, fista_eps=1e2, nforhc=10, verbose=True):
    '''  FISTA Kirchhoff-based inversion for microseismic source location

    Parameters
    ----------
    Op - PyLops Kirchhoff operator
    data - seismic data [nr,nt]
    n_xyz - dimensions of subsurface model/migrated image
    niter - number of iterations for inversion, default 100
    nforhc - number of points for hypocenter, default 10
    verbose - run in verbose mode, default True

    Returns
    -------
    fista_mig - FISTA inversion volume
    hc - determined hypocentral location

    '''
    nx, ny, nz = n_xyz
    with pylops.disabled_ndarray_multiplication():
        fista_mig = fista(Op, data.flatten(), niter=niter, eps=fista_eps, show=verbose)[0].reshape(nx, ny, nz)
    hc, _ = get_max_locs(fista_mig, n_max=nforhc, rem_edge=False)
    return fista_mig, hc

def diffraction_stacking(tt, data, dt, n_xyz, nforhc=10, verbose=True):
    '''  Diffraction stacking for microseismic source location

    Parameters
    ----------
    tt - traveltimes array [nr,nx,ny,nz]
    data - seismic data [nr,nt]
    n_xyz - dimensions of subsurface model/migrated image
    nforhc - number of points for hypocenter, default 10
    verbose - run in verbose mode, default True

    Returns
    -------
    ds_im_vol - image volume after diffraction stacking
    hc - determined hypocentral location

    '''
    # Get sizes
    nx, ny, nz = n_xyz
    ngrid = nx*ny*nz
    nr = tt.shape[0]

    # Reshape tt array
    ttg = tt.reshape(nr, -1)

    # Initialise ds image array
    ds_im = np.zeros(ngrid)

    # Loop over grid points
    for igrid in range(ngrid):
        #igrid = ix * (ny * nz) + iy * nz + iz
        # Get traveltime curve
        tcurve = ttg[:,igrid]
        # Get minimum time
        tmin = np.min(tcurve)
        # Get shifts
        itshifts =  np.round((tcurve - tmin)/dt)
        #tcurve = tcurve - tmin
        # Perform moveout correction for data
        data_mc = moveout_correction(data=data,itshifts=itshifts)

        ds_im[igrid] = np.max(semblance_stack(data_mc))

    ds_im_vol = ds_im.reshape(nx, ny, nz)
    
    hc, _ = get_max_locs(ds_im_vol, n_max=nforhc, rem_edge=False)
    return ds_im_vol, hc
