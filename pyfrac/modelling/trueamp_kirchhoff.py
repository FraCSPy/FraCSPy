__all__ = ["Kirchhoff"]


import logging
import os
import warnings
from typing import Optional, Tuple, Union

import numpy as np

from pylops import LinearOperator
from pylops.signalprocessing import Convolve1D
from pylops.utils import deps
from pylops.utils._internal import _value_or_sized_to_array
from pylops.utils.decorators import reshaped
from pylops.utils.tapers import taper
from pylops.utils.typing import DTypeLike, NDArray

skfmm_message = deps.skfmm_import("the kirchhoff module")
jit_message = deps.numba_import("the kirchhoff module")

if skfmm_message is None:
    import skfmm

if jit_message is None:
    from numba import jit, prange

    # detect whether to use parallel or not
    numba_threads = int(os.getenv("NUMBA_NUM_THREADS", "1"))
    parallel = True if numba_threads != 1 else False
else:
    prange = range

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class Kirchhoff(LinearOperator):
    r"""Kirchhoff Demigration operator.

    Kirchhoff-based demigration/migration operator. Uses a high-frequency
    approximation of  Green's function propagators based on ``trav``.

    Parameters
    ----------
    z : :obj:`numpy.ndarray`
        Depth axis
    x : :obj:`numpy.ndarray`
        Spatial axis
    t : :obj:`numpy.ndarray`
        Time axis for data
    srcs : :obj:`numpy.ndarray`
        Sources in array of size :math:`\lbrack 2 (3) \times n_s \rbrack`
        The first axis should be ordered as (``y``,) ``x``, ``z``.
    recs : :obj:`numpy.ndarray`
        Receivers in array of size :math:`\lbrack 2 (3) \times n_r \rbrack`
        The first axis should be ordered as (``y``,) ``x``, ``z``.
    vel : :obj:`numpy.ndarray` or :obj:`float`
        Velocity model of size :math:`\lbrack (n_y\,\times)\; n_x
        \times n_z \rbrack` (or constant)
    wav : :obj:`numpy.ndarray`
        Wavelet.
    wavcenter : :obj:`int`
        Index of wavelet center
    y : :obj:`numpy.ndarray`
        Additional spatial axis (for 3-dimensional problems)
    mode : :obj:`str`, optional
        Computation mode (``analytic``, ``eikonal`` or ``byot``, see Notes for
        more details)
    wavfilter : :obj:`bool`, optional
        .. versionadded:: 2.0.0

        Apply wavelet filter (``True``) or not (``False``)
    dynamic : :obj:`bool`, optional
        .. versionadded:: 2.0.0

        Include dynamic weights in computations (``True``) or not (``False``).
        Note that when ``mode=byot``, the user is required to provide such weights
        in ``amp``.
    trav : :obj:`numpy.ndarray` or :obj:`tuple`, optional
        Traveltime table of size
        :math:`\lbrack (n_y) n_x n_z \times n_s n_r \rbrack` or pair of traveltime tables
        of size :math:`\lbrack (n_y) n_x n_z \times n_s \rbrack` and :math:`\lbrack (n_y) n_x n_z \times n_r \rbrack`
        (to be provided if ``mode='byot'``). Note that the latter approach is recommended as less memory demanding
        than the former.
    amp : :obj:`numpy.ndarray`, optional
        .. versionadded:: 2.0.0

        Amplitude table of size
        :math:`\lbrack (n_y) n_x n_z \times n_s n_r \rbrack` or pair of amplitude tables
        of size :math:`\lbrack (n_y) n_x n_z \times n_s \rbrack` and :math:`\lbrack (n_y) n_x n_z \times n_r \rbrack`
        (to be provided if ``mode='byot'``). Note that the latter approach is recommended as less memory demanding
        than the former.
    aperture : :obj:`float` or :obj:`tuple`, optional
        .. versionadded:: 2.0.0

        Maximum allowed aperture expressed as the ratio of offset over depth. If ``None``,
        no aperture limitations are introduced. If scalar, a taper from 80% to 100% of
        aperture is applied. If tuple, apertures below the first value are
        accepted and those after the second value are rejected. A tapering is implemented
        for those between such values.
    angleaperture : :obj:`float` or :obj:`tuple`, optional
        .. versionadded:: 2.0.0

        Maximum allowed angle (either source or receiver side) in degrees. If ``None``,
        angle aperture limitations are not introduced. See ``aperture`` for implementation
        details regarding scalar and tuple cases.

    anglerefl : :obj:`np.ndarray`, optional
        .. versionadded:: 2.0.0

        Angle between the normal of the reflectors and the vertical axis in degrees
    snell : :obj:`float` or :obj:`tuple`, optional
        .. versionadded:: 2.0.0

        Threshold on Snell's law evaluation. If larger, the source-receiver-image
        point is discarded. If ``None``, no check on the validity of the Snell's
        law is performed.  See ``aperture`` for implementation details regarding
        scalar and tuple cases.
    engine : :obj:`str`, optional
        Engine used for computations (``numpy`` or ``numba``).
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        .. versionadded:: 2.0.0

        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Raises
    ------
    NotImplementedError
        If ``mode`` is neither ``analytic``, ``eikonal``, or ``byot``

    Notes
    -----
    The Kirchhoff demigration operator synthesizes seismic data given a
    propagation velocity model :math:`v` and a reflectivity model :math:`m`.
    In forward mode [1]_, [2]_:

    .. math::
        d(\mathbf{x_r}, \mathbf{x_s}, t) =
        \widetilde{w}(t) * \int_V G(\mathbf{x_r}, \mathbf{x}, t)
        m(\mathbf{x}) G(\mathbf{x}, \mathbf{x_s}, t)\,\mathrm{d}\mathbf{x}

    where :math:`m(\mathbf{x})` represents the reflectivity
    at every location in the subsurface, :math:`G(\mathbf{x}, \mathbf{x_s}, t)`
    and :math:`G(\mathbf{x_r}, \mathbf{x}, t)` are the Green's functions
    from source-to-subsurface-to-receiver and finally :math:`\widetilde{w}(t)` is
    a filtered version of the wavelet :math:`w(t)` [3]_ (or the wavelet itself when
    ``wavfilter=False``). In our implementation, the following high-frequency
    approximation of the Green's functions is adopted:

    .. math::
        G(\mathbf{x_r}, \mathbf{x}, \omega) = a(\mathbf{x_r}, \mathbf{x})
            e^{j \omega t(\mathbf{x_r}, \mathbf{x})}

    where :math:`a(\mathbf{x_r}, \mathbf{x})` is the amplitude and
    :math:`t(\mathbf{x_r}, \mathbf{x})` is the traveltime. When ``dynamic=False`` the
    amplitude is disregarded leading to a kinematic-only Kirchhoff operator.

    .. math::
        d(\mathbf{x_r}, \mathbf{x_s}, t) =
        \tilde{w}(t) * \int_V e^{j \omega (t(\mathbf{x_r}, \mathbf{x}) +
        t(\mathbf{x}, \mathbf{x_s}))} m(\mathbf{x}) \,\mathrm{d}\mathbf{x}

    On the  other hand, when ``dynamic=True``, the amplitude scaling is defined as
    :math:`a(\mathbf{x}, \mathbf{y})=\frac{1}{\|\mathbf{x} - \mathbf{y}\|}`,
    that is, the reciprocal of the distance between the two points,
    approximating the geometrical spreading of the wavefront.
    Moreover an angle scaling is included in the modelling operator
    added as follows:

    .. math::
        d(\mathbf{x_r}, \mathbf{x_s}, t) =
        \tilde{w}(t) * \int_V a(\mathbf{x}, \mathbf{x_s}) a(\mathbf{x}, \mathbf{x_r})
        \frac{|cos \theta_s + cos \theta_r|} {v(\mathbf{x})} e^{j \omega (t(\mathbf{x_r}, \mathbf{x}) +
         t(\mathbf{x}, \mathbf{x_s}))} m(\mathbf{x}) \,\mathrm{d}\mathbf{x}

    where :math:`\theta_s` and :math:`\theta_r` are the angles between the source-side
    and receiver-side rays and the normal to the reflector  at the image point (or
    the vertical axis at the image point when ``reflslope=None``), respectively.

    Depending on the choice of ``mode`` the traveltime and amplitude of the Green's
    function will be also computed differently:

    * ``mode=analytic`` or ``mode=eikonal``: traveltimes, geometrical spreading, and angles
      are computed for every source-image point-receiver triplets and the
      Green's functions are implemented from traveltime look-up tables, placing
      scaled reflectivity values at corresponding source-to-receiver time in the
      data.
    * ``byot``: bring your own tables. Traveltime table are provided
      directly by user using ``trav`` input parameter. Similarly, in this case one
      can provide their own amplitude scaling ``amp`` (which should include the angle
      scaling too).

    Three aperture limitations have been also implemented as defined by:

    * ``aperture``: the maximum allowed aperture is expressed as the ratio of
      offset over depth. This aperture limitation avoid including grazing angles
      whose contributions can introduce aliasing effects. A taper is added at the
      edges of the aperture;
    * ``angleaperture``: the maximum allowed angle aperture is expressed as the
      difference between the incident or emerging angle at every image point and
      the vertical axis (or the normal to the reflector if ``anglerefl`` is provided.
      This aperture limitation also avoid including grazing angles whose contributions
      can introduce aliasing effects. Note that for a homogenous medium and slowly varying
      heterogenous medium the offset and angle aperture limits may work in the same way;
    * ``snell``: the maximum allowed snell's angle is expressed as the absolute value of
      the sum between incident and emerging angles defined as in the ``angleaperture`` case.
      This aperture limitation is introduced to turn a scattering-based Kirchhoff engine into
      a reflection-based Kirchhoff engine where each image point is not considered as scatter
      but as a local horizontal (or straight) reflector.

    Finally, the adjoint of the demigration operator is a *migration* operator which
    projects data in the model domain creating an image of the subsurface
    reflectivity.

    .. [1] Bleistein, N., Cohen, J.K., and Stockwell, J.W..
       "Mathematics of Multidimensional Seismic Imaging, Migration and
       Inversion", 2000.

    .. [2] Santos, L.T., Schleicher, J., Tygel, M., and Hubral, P.
       "Seismic modeling by demigration", Geophysics, 65(4), pp. 1281-1289, 2000.

    .. [3] Safron, L. "Multicomponent least-squares Kirchhoff depth migration",
       MSc Thesis, 2018.

    """

    def __init__(
        self,
        z: NDArray,
        x: NDArray,
        t: NDArray,
        srcs: NDArray,
        recs: NDArray,
        wav: NDArray,
        wavcenter: int,
        y: Optional[NDArray] = None,
        wavfilter: bool = False,
        trav: Optional[NDArray] = None,
        amp: Optional[NDArray] = None,
        engine: str = "numpy",
        dtype: DTypeLike = "float64",
        name: str = "K",
    ) -> None:
        warnings.warn(
            "A new implementation of Kirchhoff is provided in v2.1.0. "
            "This currently affects only the inner working of the "
            "operator, end-users can continue using the operator in "
            "the same way. Nevertheless, it is now recommended to provide"
            "the variables trav (and amp) as a tuples containing the "
            "traveltime (and amplitude) tables for sources and receivers "
            "separately. This behaviour will eventually become default in "
            "version v3.0.0.",
            FutureWarning,
        )
        # identify geometry
        (
            self.ndims,
            _,
            dims,
            self.ny,
            self.nx,
            self.nz,
            ns,
            nr,
            _,
            _,
            _,
            _,
            _,
        ) = Kirchhoff._identify_geometry(z, x, srcs, recs, y=y)
        self.dt = t[1] - t[0]
        self.nt = len(t)

        # compute traveltime
        self.trav = trav
        self.amp = amp


        # create wavelet operator
        if wavfilter:
            self.wav = self._wavelet_reshaping(
                wav, self.dt, srcs.shape[0], recs.shape[0], self.ndims
            )
        else:
            self.wav = wav
        self.cop = Convolve1D(
            (ns * nr, self.nt), h=self.wav, offset=wavcenter, axis=1, dtype=dtype
        )

        # dimensions
        self.ns, self.nr = ns, nr
        self.nsnr = nr
        self.ni = np.prod(dims)
        dims = tuple(dims) if self.ndims == 2 else (dims[0] * dims[1], dims[2])
        dimsd = (nr, self.nt)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)

        self._register_multiplications(engine)

    @staticmethod
    def _identify_geometry(
        z: NDArray,
        x: NDArray,
        srcs: NDArray,
        recs: NDArray,
        y: Optional[NDArray] = None,
    ) -> Tuple[
        int,
        int,
        NDArray,
        int,
        int,
        int,
        int,
        int,
        float,
        float,
        float,
        NDArray,
        NDArray,
    ]:
        """Identify geometry and acquisition size and sampling"""
        ns, nr = srcs.shape[1], recs.shape[1]
        nz, nx = len(z), len(x)
        dz = np.abs(z[1] - z[0])
        dx = np.abs(x[1] - x[0])
        if y is None:
            ndims = 2
            shiftdim = 0
            ny = 1
            dy = None
            dims = np.array([nx, nz])
            dsamp = np.array([dx, dz])
            origin = np.array([x[0], z[0]])
        else:
            ndims = 3
            shiftdim = 1
            ny = len(y)
            dy = np.abs(y[1] - y[0])
            dims = np.array([ny, nx, nz])
            dsamp = np.array([dy, dx, dz])
            origin = np.array([y[0], x[0], z[0]])
        return ndims, shiftdim, dims, ny, nx, nz, ns, nr, dy, dx, dz, dsamp, origin


    def _wavelet_reshaping(
        self,
        wav: NDArray,
        dt: float,
        dimsrc: int,
        dimrec: int,
        dimv: int,
    ) -> NDArray:
        """Apply wavelet reshaping as from theory in [1]_"""
        f = np.fft.rfftfreq(len(wav), dt)
        W = np.fft.rfft(wav, n=len(wav))
        if dimsrc == 2 and dimv == 2:
            # 2D
            Wfilt = W * (2 * np.pi * f)
        elif (dimsrc == 2 or dimrec == 2) and dimv == 3:
            # 2.5D
            raise NotImplementedError("2.D wavelet currently not available")
        elif dimsrc == 3 and dimrec == 3 and dimv == 3:
            # 3D
            Wfilt = W * (-1j * 2 * np.pi * f)
        wavfilt = np.fft.irfft(Wfilt, n=len(wav))
        return wavfilt


    @staticmethod  # MODELLING
    def _travsrcrec_kirch_matvec(
        x: NDArray,
        y: NDArray,
        nr: int,
        nt: int,
        ni: int,
        dt: float,
        trav_recs: NDArray,
        amp_recs: NDArray,
    ) -> NDArray:
        for irec in range(nr):
            trav = trav_recs[:, irec]
            amp = amp_recs[:, irec]
            itrav = (trav / dt).astype("int32")
            travd = trav / dt - itrav

            for ii in range(ni):
                itravii = itrav[ii]
                travdii = travd[ii]
                ampii = amp[ii]
                if 0 <= itravii < nt - 1:
                    y[irec, itravii] += (x[ii] * ampii) * (1 - travdii)
                    y[irec, itravii + 1] += (x[ii] * ampii) * travdii
        return y

    @staticmethod  # MIGRATION
    def _travsrcrec_kirch_rmatvec(
        x: NDArray,
        y: NDArray,
        nr: int,
        nt: int,
        ni: int,
        dt: float,
        trav_recs: NDArray,
        amp_recs: NDArray,
    ) -> NDArray:
        for ii in prange(ni):
            trav_recsii = trav_recs[ii]
            amp_recsii = amp_recs[ii]
            for irec in range(nr):
                travii = trav_recsii[irec]
                ampii = amp_recsii[irec]

                itravii = int(travii / dt)
                travdii = travii / dt - itravii

                if 0 <= itravii < nt - 1:
                    y[ii] += (
                            (x[irec, itravii] * ampii) * (1 - travdii)
                            + (x[irec, itravii + 1] * ampii) * travdii
                    )
        return y

    def _register_multiplications(self, engine: str) -> None:
        if engine not in ["numpy", "numba"]:
            raise KeyError("engine must be numpy or numba")
        if engine == "numba" and jit is not None:
            # numba
            numba_opts = dict(
                nopython=True, nogil=True, parallel=parallel
            )  # fastmath=True,

            self._kirch_matvec = jit(**numba_opts)(self._travsrcrec_kirch_matvec)
            self._kirch_rmatvec = jit(**numba_opts)(self._travsrcrec_kirch_rmatvec)

        else:
            if engine == "numba" and jit is None:
                logging.warning(jit_message)

            self._kirch_matvec = self._travsrcrec_kirch_matvec
            self._kirch_rmatvec = self._travsrcrec_kirch_rmatvec

    @reshaped
    def _matvec(self, x: NDArray) -> NDArray:
        y = np.zeros((self.nr, self.nt), dtype=self.dtype)

        inputs = (
            x.ravel(),
            y,
            self.nr,
            self.nt,
            self.ni,
            self.dt,
            self.trav,
            self.amp,
        )

        y = self._kirch_matvec(*inputs)
        y = self.cop._matvec(y.ravel())
        return y

    @reshaped
    def _rmatvec(self, x: NDArray) -> NDArray:
        x = self.cop._rmatvec(x.ravel())
        x = x.reshape(self.nr, self.nt)
        y = np.zeros(self.ni, dtype=self.dtype)

        inputs = (
            x,
            y,
            self.nr,
            self.nt,
            self.ni,
            self.dt,
            self.trav,
            self.amp,
        )

        y = self._kirch_rmatvec(*inputs)
        return y