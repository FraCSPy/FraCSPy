import numpy as np


def collect_source_angles(x,y,z, reclocs, nc=3):
    '''
    cosine_theta_x = r[0]/np.sqrt(r[0]**2+r[1]**2+r[2]**2)
    cosine_theta_y = r[1]/np.sqrt(r[0]**2+r[1]**2+r[2]**2)
    cosine_theta_z = r[2]/np.sqrt(r[0]**2+r[1]**2+r[2]**2)

    Parameters
    ----------
    x - np.array of subsurface x-grid points
    y - np.array of subsurface x-grid points
    z - np.array of subsurface x-grid points
    reclocs - [3,nr]
    nc - number of components, xyz=3!

    Returns
    -------
    gamma_sourceangles  - np.array of cosine source angles [nc, nr, nx, ny, nz]
    dist_table - np.array table of euclidean distance from receivers to subsurface points [nr, nx, ny, nz]
    '''
    # quickly define number of things
    nr = reclocs.shape[1]
    nx, ny, nz = len(x), len(y), len(z)
    # Initialise outputs as empty arrays
    gamma_sourceangles = np.zeros([nc, nr, nx, ny, nz])
    dist_table = np.zeros([nr, nx, ny, nz])

    for irec in range(nr):
        # Separate receiver and source co-ords into x-,y-,z-components
        rx, ry, rz = reclocs[0, irec], reclocs[1, irec], reclocs[2, irec]
        s_locs_x, s_locs_y, s_locs_z = np.meshgrid(x, y, z, indexing='ij')

        # Compute 'distances'
        delta_x = s_locs_x - rx
        delta_y = s_locs_y - ry
        delta_z = s_locs_z - rz
        total_distance = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
        dist_table[irec] = total_distance

        # cosine x
        gamma_sourceangles[0, irec] = delta_x / total_distance
        # cosine y
        gamma_sourceangles[1, irec] = delta_y / total_distance
        # cosine z
        gamma_sourceangles[2, irec] = delta_z / total_distance

        # At rec loc total distance is zero < CANT DIVIDE BY ZERO!
        rl = np.argwhere(total_distance == 0)
        if len(rl) > 0:
            gamma_sourceangles[:, irec, rl[0][0], rl[0][1], rl[0][2]] = 0

    return gamma_sourceangles, dist_table


def pwave_zcomp_Greens(gamma_sourceangles,
                       dist_table,
                       sloc_ind,
                       vel,
                       MT_comp_dict,
                       omega_p):

    nr = gamma_sourceangles.shape[1]

    # SLICE ON SOURCE LOC
    dist_sloc = dist_table[:, sloc_ind[0], sloc_ind[1], sloc_ind[2]]
    gamma_sloc = gamma_sourceangles[:, :, sloc_ind[0], sloc_ind[1], sloc_ind[2]]

    # INITIALISE G
    G_z = np.zeros([6, nr])  # Z-component

    all_scaler = omega_p / (4 * np.pi * np.mean(vel) ** 3)
    zcomp_gamma_ind = 2  # Gamma index related to velocity direction
    for irec in range(nr):
        for cmp_dict in MT_comp_dict:
            el_indic = cmp_dict['elementID']
            p_gamma_ind, q_gamma_ind = cmp_dict['pq']

            gamma_elements = gamma_sloc[zcomp_gamma_ind,irec] * gamma_sloc[p_gamma_ind,irec] * gamma_sloc[q_gamma_ind,irec]

            G_z[el_indic, irec] = all_scaler * gamma_elements * dist_sloc[irec]**-1

    return G_z

def pwave_Greens_comp(gamma_sourceangles,
                      dist_table,
                      sloc_ind,
                      vel,
                      MT_comp_dict,
                      comp_gamma_ind,
                       omega_p,
                      ):
    '''

    Parameters
    ----------
    gamma_sourceangles
    dist_table: :obj:`numpy.ndarray`
        Travel distance table from all possible source locs (reference grid) to receivers,
    sloc_ind: :list
        Index of source location, [sxi, syi, szi], wrt reference grid indices
    omega_p
    vel
    MT_comp_dict
    comp_gamma_ind : :obj:`numpy.ndarray`
        Gamma index related to velocity direction, 0=x, 1=y, 2=z

    Returns
    -------

    '''
    nr = gamma_sourceangles.shape[1]

    # SLICE ON SOURCE LOC
    dist_sloc = dist_table[:, sloc_ind[0], sloc_ind[1], sloc_ind[2]]
    gamma_sloc = gamma_sourceangles[:, :, sloc_ind[0], sloc_ind[1], sloc_ind[2]]

    # INITIALISE G
    G = np.zeros([6, nr])  # Z-component
    all_scaler = omega_p / (4 * np.pi * np.mean(vel) ** 3)

    for irec in range(nr):
        for cmp_dict in MT_comp_dict:
            el_indic = cmp_dict['elementID']
            p_gamma_ind, q_gamma_ind = cmp_dict['pq']

            gamma_elements = gamma_sloc[comp_gamma_ind,irec] * gamma_sloc[p_gamma_ind,irec] * gamma_sloc[q_gamma_ind,irec]

            G[el_indic, irec] = all_scaler * gamma_elements * dist_sloc[irec]**-1

    return G

def multicomp_Greens_Pwave(nxyz,
                           nr,
                           gamma_sourceangles,
                           dist_table,
                           vel,
                           MT_comp_dict,
                           omega_p,
                           ):

    nx,ny,nz = nxyz

    Gx = np.zeros([6, nr, nx, ny, nz])
    Gy = np.zeros([6, nr, nx, ny, nz])
    Gz = np.zeros([6, nr, nx, ny, nz])

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                Gx[:, :, ix, iy, iz] = pwave_Greens_comp(gamma_sourceangles,
                                                             dist_table,
                                                             [ix, iy, iz],
                                                             vel,
                                                             MT_comp_dict,
                                                             comp_gamma_ind=0,
                                                             omega_p=omega_p)

                Gy[:, :, ix, iy, iz] = pwave_Greens_comp(gamma_sourceangles,
                                                             dist_table,
                                                             [ix, iy, iz],
                                                             vel,
                                                             MT_comp_dict,
                                                             comp_gamma_ind=1,
                                                             omega_p=omega_p)

                Gz[:, :, ix, iy, iz] = pwave_Greens_comp(gamma_sourceangles,
                                                             dist_table,
                                                             [ix, iy, iz],
                                                             vel,
                                                             MT_comp_dict,
                                                             comp_gamma_ind=2,
                                                             omega_p=omega_p)
    return Gx, Gy, Gz


def singlecomp_Greens_Pwave(nxyz,
                            nr,
                            gamma_sourceangles,
                            dist_table,
                            vel,
                            MT_comp_dict,
                            omega_p,
                            ):

    nx,ny,nz = nxyz
    Gz = np.zeros([6, nr, nx, ny, nz])

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                Gz[:, :, ix, iy, iz] = pwave_Greens_comp(gamma_sourceangles,
                                                         dist_table,
                                                         [ix, iy, iz],
                                                         vel,
                                                         MT_comp_dict,
                                                         comp_gamma_ind=2,
                                                         omega_p=omega_p)
    return Gz
