__all__ = [
    "MTInversion",
]


from fracspy.mtinversion.greensfunction import *
from fracspy.mtinversion.mtai import mtamplitude_inversion
from fracspy.mtinversion.mtwi import lsqr_mtsolver
from fracspy.mtinversion.utils import get_mt_computation_dict

_mt_kind = {"ai": mtamplitude_inversion,
            "wi": lsqr_mtsolver,
            }


class MTInversion():
    """Moment-Tensor inversion

    This class acts as an abstract interface for users to perform
    moment-tensor inversion location on a microseismic dataset

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        X-axis
    y : :obj:`numpy.ndarray`
        Y-axis
    z : :obj:`numpy.ndarray`
        Z-axis
    z : :obj:`numpy.ndarray`
        Z-axis
    recs : :obj:`numpy.ndarray`
        Receiver locations of size :math:`3 \times n_r`
    src_idx : :obj:`numpy.ndarray`
        Source location indices (relative to x, y, and z axes)
    vel : :obj:`numpy.ndarray`
        Velocity model of size :math:`n_x \times n_y \times n_z`
    omega_p : :obj:`float`
        Central frequency of source
    vel : :obj:`numpy.ndarray`
        Velocity model of size :math:`n_x \times n_y \times n_z`

    """
    def __init__(self, x, y, z, recs, vel, omega_p):
        self.x, self.y, self.z = x, y, z
        self.n_xyz = x.size, y.size, z.size
        self.recs = recs
        self.vel = vel
        self.omega_p = omega_p
        self._precompute()

    def _precompute(self):
        """Pre-computations

        Pre-compute parameters that can be re-used for different source locations
        """
        self.cosine_sourceangles, self.dists = collect_source_angles(self.x, self.y, self.z, recs=self.recs) #, nc=3)

    def apply(self, data, src_idx,  kind="ai", **kwargs):
        """Perform MT inversion

        This method performs MT inversion location for the provided dataset (either amplitudes
        as function of receiver location or waveforms) and source location (previously estimated via, e.g.
        `fracspy.location.Location`) from the pre-defined acquisition geometry using one of the available
        inversion techniques.

        .. note:: This method can be called multiple times using different input datasets and source locations
          and/or imaging methods as the internal parameters are not modified during the
          location procedure.

        """
        Gz = pwave_Greens_comp(self.cosine_sourceangles,
                               self.dists,
                               src_idx,
                               self.vel,
                               get_mt_computation_dict(),
                               comp_gamma_ind=2,
                               omega_p=self.omega_p,
                               )

        return _mt_kind[kind](Gz, data, **kwargs)
