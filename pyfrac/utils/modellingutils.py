import numpy as np
import os


def make_homo_model(dimlims,
                    deltadims,
                    subsurface_properties,
                    pad=True,
                    npad=30,
                    freesurface=True
                    ):
    """ Make a 3D homogeneous velocity model

    Parameters
    ----------
    dimlims : :obj:`float` or :obj:`tuple`
        Assuming a starting point of zero, the maximum offset in x,y,z IN METERS, if given as float the limits are
        spread across the 3 dimensions
    deltadims : :obj:`float` or :obj:`tuple`
        Step in each axis (i.e., dx,dy,dz) IN METERS, if given as float the limits are
    subsurface_properties : :obj:`tuple`
        [vp, vs, rho]
    pad : :obj:`bool`, optional
        pad model to accomodate for absorbing boundaries, default True
    npad : :obj:`int`, optional
        number of grid points to add for absorbing boundaries, default 30
    freesurface : :obj:`bool`, optional
        remove padding from top of model, default True

    Returns
    -------

    """
    # Assign variables
    xmax = ymax = zmax = dimlims  # m
    dx = dy = dz = deltadims  # m

    # Compute number of grid points
    nx = ny = nz = int(xmax / dx)  # Grid points
    nx_m = ny_m = nz_m = nx

    if pad:
        nx_m = ny_m = nz_m = int(nx + npad * 2)  # Grid points

    if freesurface:
        ny_m = int(ny + npad)

    # Make models
    vp = subsurface_properties[0]  # m/s
    vs = subsurface_properties[1]  # m/s
    rho = subsurface_properties[2];  # kg/m3

    mod_vp = vp * np.ones([nx_m, ny_m, nz_m])
    mod_vs = vs * np.ones([nx_m, ny_m, nz_m])
    mod_rho = rho * np.ones([nx_m, ny_m, nz_m])

    return mod_vp, mod_vs, mod_rho


def comp_source_loc(orig_xyz,
                    deltadims,
                    npad=30,
                    freesurface=True):
    """ convert orig source xyz to new xyz taking into account padding

    Parameters
    ----------
    orig_xyz
    deltadims
    npad
    freesurface

    Returns
    -------

    """

    dx = dy = dz = deltadims  # m

    # convert to gridpoint
    o_source_gp = [x / dx for x in orig_xyz]
    n_source_gp = [x + npad for x in o_source_gp]

    # Account for freesurface
    if freesurface:
        n_source_gp[1] = o_source_gp[1]

    # Convert back to meters
    n_source_xyz = [x * dx for x in n_source_gp]

    return n_source_xyz


def get_recs_on_face(nrpF, nx, dx, rec_buffer=6, npad=30, freesurface=True):
    """ Taking into account padding and free surface

    Parameters
    ----------
    nrpF : :obj:`int`
        number of receivers per face
    nx
    dx
    rec_buffer
    npad
    freesurface

    Returns
    -------

    """
    # Get grid location without considering padding
    start_point = rec_buffer * dx
    end_point = (nx - (2 * rec_buffer)) * dx
    grid_locs = np.linspace(start_point, end_point, nrpF)

    if freesurface:
        ypad = 0
    else:
        ypad = npad

    if npad == 0:
        xstrt0 = ystrt0 = zstrt0 = dx
        xstrt1 = ystrt1 = zstrt1 = (nx - 1) * dx
    else:
        xstrt0 = zstrt0 = npad * dx
        xstrt1 = zstrt1 = (npad + nx) * dx
        ystrt0 = ypad * dx
        ystrt1 = (ypad + nx) * dx

    # X FACE
    F0_rx, F0_ry, F0_rz = np.meshgrid(xstrt0, grid_locs + (ypad * dx), grid_locs + (npad * dx))  # X face 0
    F1_rx, F1_ry, F1_rz = np.meshgrid(xstrt1, grid_locs + (ypad * dx), grid_locs + (npad * dx))  # X face 1
    # Y FACE
    F2_rx, F2_ry, F2_rz = np.meshgrid(grid_locs + (npad * dx), ystrt0, grid_locs + (npad * dx))  # Y face 0
    F3_rx, F3_ry, F3_rz = np.meshgrid(grid_locs + (npad * dx), ystrt1, grid_locs + (npad * dx))  # Y face 1
    # Z FACE
    F4_rx, F4_ry, F4_rz = np.meshgrid(grid_locs + (npad * dx), grid_locs + (ypad * dx), zstrt0)  # Z face 0
    F5_rx, F5_ry, F5_rz = np.meshgrid(grid_locs + (npad * dx), grid_locs + (ypad * dx), zstrt1)  # Z face 1

    # COMBINE
    grid_rx = np.vstack([rx.flatten() for rx in [F0_rx, F1_rx, F2_rx, F3_rx, F4_rx, F5_rx, ]])
    grid_ry = np.vstack([ry.flatten() for ry in [F0_ry, F1_ry, F2_ry, F3_ry, F4_ry, F5_ry, ]])
    grid_rz = np.vstack([rz.flatten() for rz in [F0_rz, F1_rz, F2_rz, F3_rz, F4_rz, F5_rz, ]])

    gridded_recs = np.vstack((grid_rx.flatten(), grid_ry.flatten(), grid_rz.flatten()))
    nr = gridded_recs.shape[1]

    return gridded_recs, nr
