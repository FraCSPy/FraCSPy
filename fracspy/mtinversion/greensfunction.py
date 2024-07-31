import numpy as np


def collect_source_angles(x, y, z, recs): #, nc=3):
    """Angles between sources and receivers

    Compute angles between sources in a regular 3-dimensional
    grid and receivers

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        X-axis
    y : :obj:`numpy.ndarray`
        Y-axis
    z : :obj:`numpy.ndarray`
        Z-axis
    recs : :obj:`numpy.ndarray`
        Receiver locations of size :math:`3 \times n_r`
    # nc : :obj:`int`, optional
    #    Number of components

    Returns
    -------
    cosine_sourceangles : :obj:`numpy.ndarray`
        Cosine source angles of size :math:`3 \times n_r \times n_x \times n_y \times n_z`
    dists : :obj:`numpy.ndarray`
        Distances of size :math:`\times n_r \times n_x \times n_y \times n_z`

    Notes
    -----

    This routine computes the cosine source angles and distances between any
    pair of sources and receivers.

    Cosine source angles are defined as follows:

    .. math::
        cos(\theta_x) = (x_s - x_r) / d \\
        cos(\theta_y) = (y_s - y_r) / d \\
        cos(\theta_z) = (z_s - z_r) / d \\

    where :math:`d=\sqrt{(x_s - x_r)^2+(y_s - y_r)^2+(z_s - z_r)^2+}` are the distances.

    """
    # Define dimensions
    nc = 3
    nr = recs.shape[1]
    nx, ny, nz = len(x), len(y), len(z)

    # Initialise angle and distance tables
    cosine_sourceangles = np.zeros([nc, nr, nx, ny, nz])
    dists = np.zeros([nr, nx, ny, nz])

    for irec in range(nr):
        # Separate receiver into x-,y-,z-components
        rx, ry, rz = recs[0, irec], recs[1, irec], recs[2, irec]

        # Create regular grid of source coordinates
        s_locs_x, s_locs_y, s_locs_z = np.meshgrid(x, y, z, indexing='ij')

        # Compute pair-wise and total distances
        delta_x = s_locs_x - rx
        delta_y = s_locs_y - ry
        delta_z = s_locs_z - rz
        total_distance = np.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

        # Assign distances for a given receiver to distance table
        dists[irec] = total_distance

        # Remove rec loc from total distance table
        rl = np.argwhere(total_distance == 0)
        if len(rl) > 0:
            # Temporarily set total_distance to 999 for rec loc
            total_distance[rl[0][0], rl[0][1], rl[0][2]] = 999

        # Compute cosines of source angles
        cosine_sourceangles[0, irec] = delta_x / total_distance
        cosine_sourceangles[1, irec] = delta_y / total_distance
        cosine_sourceangles[2, irec] = delta_z / total_distance
        if len(rl) > 0:
            # Put correct source angle for receiver location
            cosine_sourceangles[:, irec, rl[0][0], rl[0][1], rl[0][2]] = 0

    return cosine_sourceangles, dists


def pwave_zcomp_Greens(cosine_sourceangles,
                       dists,
                       src_idx,
                       vel,
                       MT_comp_dict,
                       omega_p):
    """P-wave, z-component Green's function

    Compute Green's function for the z-component of the P-wave
    between sources in a regular 3-dimensional
    grid and receivers

    Parameters
    ----------
    cosine_sourceangles : :obj:`numpy.ndarray`
        Cosine source angles of size :math:`3 \times n_r \times n_x \times n_y \times n_z`
    dists : :obj:`numpy.ndarray`
        Distances of size :math:`\times n_r \times n_x \times n_y \times n_z`
    src_idx : :obj:`numpy.ndarray`
        Source location indices (relative to x, y, and z axes)

    Returns
    -------

    Notes
    -----
    This methodc computes the amplitudes of z-component of the Green's functions associated to the first P-wave arrival,
    assuming a known source location based on the far-field particle velocity expression from a moment tensor source
    in a homogeneous full space (eq. 4.29, [1]_):

    .. math::
        v_z^P = j \omega_P \left( \frac{\gamma_z\gamma_p\gamma_q}{4\pi\rho\alpha^3}  \frac{1}{r} \right) M_{pq}

    where:

    - :math:`v` is the particle velocity measurements (seismic data) at the arrival of the wave, in other words
      the P-wave peak amplitudes;

    - :math:`M` is the moment tensor;

    - :math:`\theta` describes whether we are utilising the P-wave information;

    - :math:`i` describes the z-component of the data, aligning with the below p,q definitions;

    - :math:`p` describes the first index of the moment tensor element;

    - :math:`q` describes the second index of the moment tensor element;

    - :math:`\omega_P` is the peak frequency of the given wave;

    - :math:`\gamma_{z/p/q}` is the take-off angle in the z/p/q-th direction
      (for a ray between the source and receiver);

    - :math:`r` is the distance between source and receiver;

    - :math:`\alpha` is the average velocity (currently we assume a homogeneous velocity);

    - :math:`\rho` is the average density;


    .. [1] Aki, K., and Richards, P. G. "Quantitative Seismology", University Science Books, 2002.

    """
    nr = cosine_sourceangles.shape[1]

    # Extract cosine source angle and distance at selected source location
    dist_sloc = dists[:, src_idx[0], src_idx[1], src_idx[2]]
    cosine_src = cosine_sourceangles[:, :, src_idx[0], src_idx[1], src_idx[2]]

    # Compute common scalar
    all_scaler = omega_p / (4 * np.pi * np.mean(vel) ** 3)

    # Gamma index related to velocity direction
    zcomp_gamma_ind = 2

    # Compute P-wave Green's function (z-component)
    G_z = np.zeros([6, nr])
    for irec in range(nr):
        for cmp_dict in MT_comp_dict:
            el_indic = cmp_dict['elementID']
            p_gamma_ind, q_gamma_ind = cmp_dict['pq']

            cosine_src_elements = cosine_src[zcomp_gamma_ind,irec] * cosine_src[p_gamma_ind,irec] * cosine_src[q_gamma_ind, irec]
            G_z[el_indic, irec] = all_scaler * cosine_src_elements * dist_sloc[irec] ** -1

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
