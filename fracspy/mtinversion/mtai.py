__all__ = [
    "MTA",
]

from scipy.linalg import lstsq

from fracspy.mtinversion.greensfunction import *
from fracspy.mtinversion.utils import MT_comp_dict


class MTA():
    r"""Moment-Tensor Amplitude modelling and inversion

    This class acts as an abstract interface for users to perform
    moment-tensor modelling of far-field amplitudes

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
        Central frequency of source
    cosine_sourceangles : :obj:`numpy.ndarray`, optional
        Cosine source angles of size :math:`3 \times n_r \times n_x \times n_y \times n_z`
    dists : :obj:`numpy.ndarray`
        Distances of size :math:`\times n_r \times n_x \times n_y \times n_z`, optional

    """
    def __init__(self, x, y, z, recs, vel, src_idx, cmp_idx, omega_p, cosine_sourceangles=None, dists=None):
        self.x, self.y, self.z = x, y, z
        self.n_xyz = x.size, y.size, z.size
        self.recs = recs
        self.vel = vel
        self.src_idx, self.cmp_idx = src_idx, cmp_idx
        self.omega_p = omega_p

        self.cosine_sourceangles, self.dists = cosine_sourceangles, dists
        if cosine_sourceangles is None and dists is None:
            self.cosine_sourceangles, self.dists = collect_source_angles(self.x, self.y, self.z, recs=self.recs)
        elif (cosine_sourceangles is None and dists is not None) or (cosine_sourceangles is not None and dists is  None):
            raise NotImplementedError('Must provide both cosine_sourceangles and dists or neither')
        self.G = self._create_greens()

    def _create_greens(self):
        return pwave_greens_comp(
            self.cosine_sourceangles,
            self.dists,
            self.src_idx,
            self.vel,
            MT_comp_dict,
            self.cmp_idx,
            self.omega_p,
        )

    def model(self, mt):
        """Modelling

        This method models the amplitudes resulting from a given moment-tensor source

        Parameters
        ----------
        mt : :obj:`numpy.ndarray`
            Moment tensor of size :math`6`

        Results
        -------
        d : :obj:`numpy.ndarray`
            Modelled amplitudes of size :math`n_r`

        """
        d = np.matmul(self.G.T, mt)
        return d

    def invert(self, d):
        """Inversion

        This method inverts the amplitudes for the corresponding given moment-tensor source

        Parameters
        ----------
        d : :obj:`numpy.ndarray`
            Amplitudes of size :math`n_r`

        Returns
        -------
        mt : :obj:`numpy.ndarray`
            Moment tensor of size :math`6`

        """
        mt = lstsq(self.G.T, d)[0]
        return mt

