import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def recgeom_rockblock(nxyz, dxyz, dr_xyz=[5], recbuf_gps=5, sofishift=1):

    # define params
    nx,ny,nz = nxyz[0], nxyz[1], nxyz[2]
    dx,dy,dz = dxyz[0], dxyz[1], dxyz[2]

    # Receiver buffer as function of grid point
    rec_buffer = recbuf_gps

    # RECEIVER GRID - ONLY ON CUBE SURFACES
    grid_z_locs = np.arange(dz+(rec_buffer*dz), (dx*nx)-(rec_buffer*dx), dr_xyz)
    grid_x_locs = np.arange(dz+(rec_buffer*dz), (dx*nx)-(rec_buffer*dx), dr_xyz)
    grid_y_locs = np.arange(dz+(rec_buffer*dz), (dx*nx)-(rec_buffer*dx), dr_xyz)

    F0_rx, F0_ry, F0_rz = np.meshgrid((sofishift*dx),
                                      grid_y_locs,
                                      grid_z_locs)      # X face 0
    F1_rx, F1_ry, F1_rz = np.meshgrid((nx*dx),
                                      grid_y_locs,
                                      grid_z_locs)      # X face 1

    F2_rx, F2_ry, F2_rz = np.meshgrid(grid_x_locs,
                                      (sofishift*dy),
                                      grid_z_locs)      # Y face 0
    F3_rx, F3_ry, F3_rz = np.meshgrid(grid_x_locs,
                                      (ny*dy),
                                      grid_z_locs)      # Y face 1

    F4_rx, F4_ry, F4_rz = np.meshgrid(grid_x_locs,
                                      grid_y_locs,
                                      (sofishift*dz))                # Z face 0
    F5_rx, F5_ry, F5_rz = np.meshgrid(grid_x_locs,
                                      grid_y_locs,
                                      (nz*dz))            # Z face 1

    grid_rx = np.vstack([rx.flatten() for rx in [F0_rx,F1_rx,F2_rx,F3_rx,F4_rx,F5_rx,]])
    grid_ry = np.vstack([ry.flatten() for ry in [F0_ry,F1_ry,F2_ry,F3_ry,F4_ry,F5_ry,]])
    grid_rz = np.vstack([rz.flatten() for rz in [F0_rz,F1_rz,F2_rz,F3_rz,F4_rz,F5_rz,]])

    gridded_recs = np.vstack((grid_rx.flatten(), grid_ry.flatten(), grid_rz.flatten()))
    nr = gridded_recs.shape[1]

    return gridded_recs, nr


def recgeom(nxyz, dxyz, nrperfc_xyz=[5], recbuf_gps=5, sofishift=1):

    # define params
    nx,ny,nz = nxyz[0], nxyz[1], nxyz[2]
    dx,dy,dz = dxyz[0], dxyz[1], dxyz[2]

    # Number of receivers per face per axis
    if len(nrperfc_xyz)>1:
        nr_z = nrperfc_xyz[2]
        nr_x = nrperfc_xyz[0]
        nr_y = nrperfc_xyz[1]
    else: nr_z = nr_x = nr_y = nrperfc_xyz[0]

    # Receiver buffer as function of grid point
    rec_buffer = recbuf_gps

    rec_string = '%i-by-%i-by-%i'%(nr_x,nr_y,nr_z)

    # RECEIVER GRID - ONLY ON CUBE SURFACES
    grid_z_locs = np.linspace(rec_buffer*dz + (sofishift*dx), (nx*dx)-(rec_buffer*dx), nr_z)
    grid_x_locs = np.linspace(rec_buffer*dx + (sofishift*dx), (nx*dx)-(rec_buffer*dx), nr_x)
    grid_y_locs = np.linspace(rec_buffer*dy + (sofishift*dx), (nx*dx)-(rec_buffer*dx), nr_y)

    F0_rx, F0_ry, F0_rz = np.meshgrid((sofishift*dx),
                                      grid_y_locs,
                                      grid_z_locs)      # X face 0
    F1_rx, F1_ry, F1_rz = np.meshgrid((nx*dx),
                                      grid_y_locs,
                                      grid_z_locs)      # X face 1

    F2_rx, F2_ry, F2_rz = np.meshgrid(grid_x_locs,
                                      (sofishift*dy),
                                      grid_z_locs)      # Y face 0
    F3_rx, F3_ry, F3_rz = np.meshgrid(grid_x_locs,
                                      (ny*dy),
                                      grid_z_locs)      # Y face 1

    F4_rx, F4_ry, F4_rz = np.meshgrid(grid_x_locs,
                                      grid_y_locs,
                                      (sofishift*dz))                # Z face 0
    F5_rx, F5_ry, F5_rz = np.meshgrid(grid_x_locs,
                                      grid_y_locs,
                                      (nz*dz))            # Z face 1

    grid_rx = np.vstack([rx.flatten() for rx in [F0_rx,F1_rx,F2_rx,F3_rx,F4_rx,F5_rx,]])
    grid_ry = np.vstack([ry.flatten() for ry in [F0_ry,F1_ry,F2_ry,F3_ry,F4_ry,F5_ry,]])
    grid_rz = np.vstack([rz.flatten() for rz in [F0_rz,F1_rz,F2_rz,F3_rz,F4_rz,F5_rz,]])

    gridded_recs = np.vstack((grid_rx.flatten(), grid_ry.flatten(), grid_rz.flatten()))
    nr = gridded_recs.shape[1]

    return gridded_recs, nr, rec_string

def recgeom_indivFaces(nxyz, dxyz, nrperfc_xyz=[5], recbuf_gps=5):

    # define params
    nx,ny,nz = nxyz[0], nxyz[1], nxyz[2]
    dx,dy,dz = dxyz[0], dxyz[1], dxyz[2]

    # Number of receivers per face per axis
    if len(nrperfc_xyz)>1:
        nr_z = nrperfc_xyz[2]
        nr_x = nrperfc_xyz[0]
        nr_y = nrperfc_xyz[1]
    else: nr_z = nr_x =nr_y = nrperfc_xyz[0]

    # Receiver buffer as function of grid point
    rec_buffer = recbuf_gps

    rec_string = '%i-by-%i-by-%i'%(nr_x,nr_y,nr_z)

    # RECEIVER GRID - ONLY ON CUBE SURFACES
    grid_z_locs = np.linspace(rec_buffer*dz, (nz-rec_buffer)*dz, nr_z)
    grid_x_locs = np.linspace(rec_buffer*dx, (nx-rec_buffer)*dx, nr_x)
    grid_y_locs = np.linspace(rec_buffer*dy, (ny-rec_buffer)*dy, nr_y)

    F0_rx, F0_ry, F0_rz = np.meshgrid(0,           grid_y_locs, grid_z_locs)      # X face 0
    F1_rx, F1_ry, F1_rz = np.meshgrid(nx*dx,       grid_y_locs, grid_z_locs)      # X face 1
    F2_rx, F2_ry, F2_rz = np.meshgrid(grid_x_locs, 0,           grid_z_locs)      # Y face 0
    F3_rx, F3_ry, F3_rz = np.meshgrid(grid_x_locs, ny*dy,       grid_z_locs)      # Y face 1
    F4_rx, F4_ry, F4_rz = np.meshgrid(grid_x_locs, grid_y_locs, 0)                # Z face 0
    F5_rx, F5_ry, F5_rz = np.meshgrid(grid_x_locs, grid_y_locs, nz*dz)            # Z face 1

    grid_rx = np.vstack([rx.flatten() for rx in [F0_rx,F1_rx,F2_rx,F3_rx,F4_rx,F5_rx,]])
    grid_ry = np.vstack([ry.flatten() for ry in [F0_ry,F1_ry,F2_ry,F3_ry,F4_ry,F5_ry,]])
    grid_rz = np.vstack([rz.flatten() for rz in [F0_rz,F1_rz,F2_rz,F3_rz,F4_rz,F5_rz,]])

    gridded_recs = np.vstack((grid_rx.flatten(), grid_ry.flatten(), grid_rz.flatten()))
    nr = gridded_recs.shape[1]

    F0 = np.vstack((F0_rx.flatten(), F0_ry.flatten(), F0_rz.flatten()))
    F1 = np.vstack((F1_rx.flatten(), F1_ry.flatten(), F1_rz.flatten()))
    F2 = np.vstack((F2_rx.flatten(), F2_ry.flatten(), F2_rz.flatten()))
    F3 = np.vstack((F3_rx.flatten(), F3_ry.flatten(), F3_rz.flatten()))
    F4 = np.vstack((F4_rx.flatten(), F4_ry.flatten(), F4_rz.flatten()))
    F5 = np.vstack((F5_rx.flatten(), F5_ry.flatten(), F5_rz.flatten()))

    return F0, F1, F2, F3, F4, F5



def dasgeom(nxyz, dxyz, nrperfc_xyz=[5], recbuf_gps=5):

    # define params
    nx,ny,nz = nxyz[0], nxyz[1], nxyz[2]
    dx,dy,dz = dxyz[0], dxyz[1], dxyz[2]

    # Number of receivers per face per axis
    if len(nrperfc_xyz)>1:
        nr_z = nrperfc_xyz[2]
        nr_x = nrperfc_xyz[0]
        nr_y = nrperfc_xyz[1]
    else: nr_z = nr_x =nr_y = nrperfc_xyz[0]

    # Receiver buffer as function of grid point
    rec_buffer = recbuf_gps

    # RECEIVER GRID - ONLY ON CUBE SURFACES
    grid_z_locs = np.linspace(rec_buffer*dz, (nz-rec_buffer)*dz, nr_z)
    grid_x_locs = np.linspace(rec_buffer*dx, (nx-rec_buffer)*dx, nr_x)
    grid_y_locs = np.linspace(rec_buffer*dy, (ny-rec_buffer)*dy, nr_y)

    F0_rx, F0_ry, F0_rz = np.meshgrid(0,           grid_y_locs, grid_z_locs)      # X face 0
    F1_rx, F1_ry, F1_rz = np.meshgrid(nx*dx,       grid_y_locs, grid_z_locs)      # X face 1
    F2_rx, F2_ry, F2_rz = np.meshgrid(grid_x_locs, 0,           grid_z_locs)      # Y face 0
    F3_rx, F3_ry, F3_rz = np.meshgrid(grid_x_locs, ny*dy,       grid_z_locs)      # Y face 1
    F4_rx, F4_ry, F4_rz = np.meshgrid(grid_x_locs, grid_y_locs, 0)                # Z face 0
    F5_rx, F5_ry, F5_rz = np.meshgrid(grid_x_locs, grid_y_locs, nz*dz)            # Z face 1

    # DAS  LINES
    # ----------------------------------------------------------------------
    #                           VIL
    # ----------------------------------------------------------------------
    VIL_rx = np.vstack([rx.flatten() for rx in [F0_rx, F1_rx, F2_rx, F3_rx, ]])
    VIL_ry = np.vstack([ry.flatten() for ry in [F0_ry, F1_ry, F2_ry, F3_ry, ]])
    VIL_rz = np.vstack([rz.flatten() for rz in [F0_rz, F1_rz, F2_rz, F3_rz, ]])
    VIL_recs = np.vstack((VIL_rx.flatten(), VIL_ry.flatten(), VIL_rz.flatten()))

    # ----------------------------------------------------------------------
    #                           VXL
    # ----------------------------------------------------------------------
    VXL_rx = np.vstack([rx.flatten() for rx in [F2_rx, F3_rx, F4_rx, F5_rx, ]])
    VXL_ry = np.vstack([ry.flatten() for ry in [F2_ry, F3_ry, F4_ry, F5_ry, ]])
    VXL_rz = np.vstack([rz.flatten() for rz in [F2_rz, F3_rz, F4_rz, F5_rz, ]])
    VXL_recs = np.vstack((VXL_rx.flatten(), VXL_ry.flatten(), VXL_rz.flatten()))

    # ----------------------------------------------------------------------
    #                           H
    # ----------------------------------------------------------------------
    H_rx = np.vstack([rx.flatten() for rx in [F0_rx, F1_rx, F4_rx, F5_rx, ]])
    H_ry = np.vstack([ry.flatten() for ry in [F0_ry, F1_ry, F4_ry, F5_ry, ]])
    H_rz = np.vstack([rz.flatten() for rz in [F0_rz, F1_rz, F4_rz, F5_rz, ]])
    H_recs = np.vstack((H_rx.flatten(), H_ry.flatten(), H_rz.flatten()))

    das_recs = np.hstack([VIL_recs, VXL_recs, H_recs])
    return das_recs


def rec2das_datatrnsfrm(data, nr, nt):
    nr_x,nr_y,nr_z = nr[0], nr[1], nr[2]
    face_data = data.reshape(6,nr_x,nr_x,nt)
    # Place into face categories
    face_groupA = face_data[:4]
    face_groupB = face_data[2:]
    face_groupC = face_data[[1,2,4,5]]

    # average over nondominant axis
    face_groupA_avg = np.repeat(np.sum(face_groupA,axis=1), nr_x, axis=1).reshape(4*nr_x*nr_x,nt)
    face_groupB_avg = np.repeat(np.sum(face_groupB,axis=1), nr_x, axis=1).reshape(4*nr_x*nr_x,nt)
    face_groupC_avg = np.repeat(np.sum(face_groupC,axis=1), nr_x, axis=1).reshape(4*nr_x*nr_x,nt)

    # stack back to 2D
    pdas_data = np.vstack([face_groupA_avg,face_groupB_avg,face_groupC_avg])

    return pdas_data


def acquisition_plot_geophones(gridded_recs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    r = [0,200]
    Z0, Z1 = 0*np.ones_like(X), 200*np.ones_like(X)
    X, Y = np.meshgrid(r, r)

    ax.plot_surface(X,Y,Z1, color='grey', alpha=0.5, zorder=0.3)
    ax.plot_surface(X,Y,Z0,color='grey',  alpha=0.5, zorder=0.3)
    ax.plot_surface(X,Z0,Y, color='grey', alpha=0.5, zorder=0.3)
    ax.plot_surface(X,Z1,Y, color='grey', alpha=0.5, zorder=0.3)
    ax.plot_surface(Z0,X,Y, color='grey', alpha=0.5, zorder=0.3)
    ax.plot_surface(Z1,X,Y, color='grey', alpha=0.5, zorder=0.3)


    # # SOURCE
    ax.scatter3D(100, 100, 100, alpha=1, color='yellow', marker='*', edgecolor='k', s=100, label='Source');

    ax.scatter3D(F2_rx, F2_ry, F2_rz,alpha=1, color='blue', marker='v', edgecolor='k', label='Point Recs');
    ax.scatter3D(F5_rx, F5_ry, F5_rz,alpha=1, color='blue', marker='v', edgecolor='k', );
    ax.scatter3D(F1_rx, F1_ry, F1_rz,alpha=1, color='blue', marker='v', edgecolor='k', );


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.suptitle('3D Velocity Model \n and Acquisition', fontsize=18)
    plt.tight_layout()