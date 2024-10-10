__all__ = [
    "MTW",
]

from scipy.sparse.linalg import lsqr
from pylops.optimization.sparsity import *

from fracspy.modelling.mt_kirchhoff import TAKirchhoff, MTSKirchhoff, MTMKirchhoff
from fracspy.mtinversion.greensfunction import *
from fracspy.mtinversion.utils import MT_comp_dict


class MTW():
    r"""Moment-Tensor Waveform modelling and inversion

    This class acts as an abstract interface for users to perform
    moment-tensor modelling of waveforms

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
    vel : :obj:`numpy.ndarray`
        Velocity model of size :math:`n_x \times n_y \times n_z`
    src_idx : :obj:`numpy.ndarray`
        Source location indices (relative to x, y, and z axes)
    comp_idx : :obj:`int`
        Index of component at receiver side
    omega_p : :obj:`float`
        Peak frequency of the given wave
    aoi : :obj:`tuple`
        Area of interest for waveform computation defined as half windows to place either size of the source in center
        of region (defined by `src_idx`)
    t : :obj:`numpy.ndarray`
        Time axis for data
    wav : :obj:`numpy.ndarray`
        Wavelet.
    wavcenter : :obj:`int`
        Index of wavelet center
    Ms_scaling : :obj:`float`
        Scaling to be incorporated in the MTI
    engine : :obj:`str`, optional
        Engine used for computations (``numpy`` or ``numba``).
    multicomp : : obj:`boolean`
        Whether running for single or multicomponent data
    cosine_sourceangles : :obj:`numpy.ndarray`
        Cosine source angles of size :math:`3 \times n_r \times n_x \times n_y \times n_z`
    dists : :obj:`numpy.ndarray`
        Distances of size :math:`\times n_r \times n_x \times n_y \times n_z`
    """
    def __init__(self, x, y, z, recs, vel, src_idx, comp_idx,
                 omega_p, aoi, t, wav, wavc,
                 Ms_scaling=1., engine="numpy", multicomp=False,
                 cosine_sourceangles=None, dists=None):
        self.x, self.y, self.z = x, y, z
        self.n_xyz = x.size, y.size, z.size
        self.recs = recs
        self.nr = recs.shape[1]
        self.vel = vel
        if isinstance(vel, np.ndarray):
            self.mode = "eikonal"
        else:
            self.mode = "analytic"
        self.src_idx = src_idx
        self.comp_idx = comp_idx
        self.omega_p = omega_p
        self.aoi = aoi

        self.t, self.wav, self.wavc = t, wav, wavc
        self.nt = t.size

        self.multicomp = multicomp
        self.ncomps = 3 if multicomp else 1
        self.Ms_scaling = Ms_scaling
        self.engine = engine

        self.cosine_sourceangles, self.dists = cosine_sourceangles, dists
        if cosine_sourceangles is None and dists is None:
            self.cosine_sourceangles, self.dists = collect_source_angles(self.x, self.y, self.z, recs=self.recs)
        elif (cosine_sourceangles is None and dists is not None) or (cosine_sourceangles is not None and dists is None):
            raise NotImplementedError('Must provide both cosine_sourceangles and dists or neither')

        self.Op, self.n_xyz_aoi = self._create_op()

    def _create_op(self):
        # Traveltime tables
        trav = TAKirchhoff._traveltime_table(self.z,
                                             self.x,
                                             y=self.y,
                                             recs=self.recs,
                                             vel=self.vel,
                                             mode=self.mode)
        trav = trav.reshape(self.n_xyz[0], self.n_xyz[1], self.n_xyz[2], self.nr).transpose([3, 0, 1, 2])

        # Define area of interest
        hwin_nx_aoi, hwin_ny_aoi, hwin_nz_aoi = self.aoi
        xsi, xfi = self.src_idx[0] - hwin_nx_aoi, self.src_idx[0] + hwin_nx_aoi + 1  # start/end index of x-region of interest
        ysi, yfi = self.src_idx[1] - hwin_ny_aoi, self.src_idx[1] + hwin_ny_aoi + 1  # start/end index of y-region of interest
        zsi, zfi = self.src_idx[2] - hwin_nz_aoi, self.src_idx[2] + hwin_nz_aoi + 1  # start/end index of z-region of interest
        nxsi = xfi - xsi
        nysi = yfi - ysi
        nzsi = zfi - zsi
        dimsai = (nxsi, nysi, nzsi)

        # Extract parameters in the area of interest
        cosine_sourceangles = self.cosine_sourceangles[:, :, xsi:xfi, ysi:yfi, zsi:zfi]
        dists = self.dists[:, xsi:xfi, ysi:yfi, zsi:zfi]
        trav = trav[:, xsi:xfi, ysi:yfi, zsi:zfi]
        nx_aoi, ny_aoi, nz_aoi = trav.shape[1:]

        # Green's functions
        if not self.multicomp:
            Gz = mt_pwave_greens_comp(n_xyz=[nx_aoi, ny_aoi, nz_aoi],
                                      cosine_sourceangles=cosine_sourceangles,
                                      dists=dists,
                                      vel=self.vel,
                                      MT_comp_dict=MT_comp_dict,
                                      comp_idx=self.comp_idx,
                                      omega_p=self.omega_p,
                                      )
            Op = MTSKirchhoff(
                self.x[xsi:xfi], self.y[ysi:yfi], self.z[zsi:zfi],
                self.recs,
                self.t,
                self.wav,
                self.wavc,
                trav,
                Gz,
                Ms_scaling=self.Ms_scaling,
                engine=self.engine,
                checkdottest=False)
        else:
            Gx, Gy, Gz = mt_pwave_greens_multicomp(n_xyz=[nx_aoi, ny_aoi, nz_aoi],
                                                   cosine_sourceangles=cosine_sourceangles,
                                                   dists=dists,
                                                   vel=self.vel,
                                                   MT_comp_dict=MT_comp_dict,
                                                   omega_p=self.omega_p,
                                                   )
            Op = MTMKirchhoff(
                self.x[xsi:xfi], self.y[ysi:yfi], self.z[zsi:zfi],
                self.recs,
                self.t,
                self.wav,
                self.wavc,
                trav,
                Gx,
                Gy,
                Gz,
                Ms_scaling=self.Ms_scaling,
                engine=self.engine,
                checkdottest=False)
        return Op, dimsai

    def model(self, mt):
        """Modelling
        """
        data = self.Op @ mt.ravel()
        data = data.reshape(self.ncomps, self.nr, self.nt).squeeze()
        return data

    def adjoint(self, data):
        """Adjoint modelling
        """
        mt = self.Op.H @ data.ravel()
        mt = mt.reshape([6, self.n_xyz_aoi[0], self.n_xyz_aoi[1], self.n_xyz_aoi[2]])
        return mt

    def lsi(self, data, niter=100, verbose=False):
        mt = lsqr(self.Op, data.ravel(), iter_lim=niter, atol=0, btol=0, show=verbose)[0]
        mt = mt.reshape([6, self.n_xyz_aoi[0], self.n_xyz_aoi[1], self.n_xyz_aoi[2]])
        return mt

    def sparselsi(self, data, niter=100, l1eps=1e2, verbose=False):
        mt = fista(
            self.Op,
            data.ravel(),
            x0=np.zeros([6, self.n_xyz_aoi[0], self.n_xyz_aoi[1], self.n_xyz_aoi[2]]).ravel(),
            niter=niter,
            eps=l1eps,
            show=verbose)[0]
        mt = mt.reshape([6, self.n_xyz_aoi[0], self.n_xyz_aoi[1], self.n_xyz_aoi[2]])
        return mt

    def invert(self, data, kind='lsi', **kwargs):
        if kind == 'adjoint':
            mt = self.adjoint(data, **kwargs)
        elif kind == 'lsi':
            mt = self.lsi(data, **kwargs)
        elif kind == 'sparselsi':
            mt = self.sparselsi(data, **kwargs)
        else:
            raise NotImplementedError('kind must be adjoint, lsi, or sparselsi')
        return mt