__all__ = [
    "MTInversion",
]


from fracspy.mtinversion.greensfunction import *
from fracspy.mtinversion.mtai import MTA
from fracspy.mtinversion.mtwi import MTW


_mt_kind = {"ai": MTA,
            "wi": MTW,
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
    recs : :obj:`numpy.ndarray`
        Receiver locations of size :math:`3 \times n_r`
    vel : :obj:`numpy.ndarray`
        Velocity model of size :math:`n_x \times n_y \times n_z`

    """
    def __init__(self, x, y, z, recs, vel):
        self.x, self.y, self.z = x, y, z
        self.n_xyz = x.size, y.size, z.size
        self.recs = recs
        self.vel = vel
        self._precompute()

    def _precompute(self):
        """Pre-computations

        Pre-compute parameters that can be re-used for different source locations
        """
        self.cosine_sourceangles, self.dists = collect_source_angles(self.x, self.y, self.z, recs=self.recs)

    def apply(self, data, src_idx, cmp_idx, omega_p, kind="ai", kwargs_engine={}, kwargs_inv={}):
        """Perform MT inversion

        This method performs MT inversion location for the provided dataset (either amplitudes
        as function of receiver location or waveforms) and source location (previously estimated via, e.g.
        `fracspy.location.Location`) from the pre-defined acquisition geometry using one of the available
        inversion techniques.

        .. note:: This method can be called multiple times using different input datasets and source locations
          and/or imaging methods as the internal parameters are not modified during the
          location procedure.

        Parameters
        ----------
        data : :obj:`numpy.ndarray`
            Amplitude data of size :math`n_r`
        src_idx : :obj:`numpy.ndarray`
            Source location indices (relative to x, y, and z axes)
        cmp_idx : :obj:`int`
            Index of component at receiver side
        omega_p : :obj:`float`
            Central frequency of source
        kind : :obj:`string`
            Type of MTI to perform, ai or wi
        kwargs_engine : dict
            Engine keywords arguments
        kwargs_inv : dict
            Inversion keywords arguments
        """
        mtengine = _mt_kind[kind](
            self.x, self.y, self.z,
            self.recs, self.vel,
            src_idx, cmp_idx,
            omega_p,
            cosine_sourceangles=self.cosine_sourceangles,
            dists=self.dists,
            **kwargs_engine)

        mt = mtengine.invert(data, **kwargs_inv)

        return mt