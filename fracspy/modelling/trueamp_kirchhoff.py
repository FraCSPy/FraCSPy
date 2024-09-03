__all__ = ["TAKirchhoff"]


import logging
import os
from typing import Optional

import numpy as np

from pylops import LinearOperator
from pylops.signalprocessing import Convolve1D
from pylops.utils import deps
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, NDArray

from fracspy.modelling.kirchhoff import Kirchhoff

jit_message = deps.numba_import("the kirchhoff module")

if jit_message is None:
    from numba import jit, prange

    # detect whether to use parallel or not
    numba_threads = int(os.getenv("NUMBA_NUM_THREADS", "1"))
    parallel = True if numba_threads != 1 else False
else:
    prange = range

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


class TAKirchhoff(LinearOperator):
    r"""True-amplitude Kirchhoff single-sided, demigration operator.

    True-amplitude Kirchhoff-based demigration/migration operator for single-sided
    propagation (from subsurface to surface). Uses a high-frequency approximation of
    Green's function propagators based on ``trav`` and  ``amp``.

    Parameters
    ----------
    z : :obj:`numpy.ndarray`
        Depth axis
    x : :obj:`numpy.ndarray`
        Spatial axis
    t : :obj:`numpy.ndarray`
        Time axis for data
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
    wavfilter : :obj:`bool`, optional
        Apply wavelet filter (``True``) or not (``False``)
    trav : :obj:`numpy.ndarray` or :obj:`tuple`, optional
        Traveltime table of size
        :math:`\lbrack (n_y) n_x n_z \times n_r \rbrack`.
    amp : :obj:`numpy.ndarray`, optional
        Amplitude table of size
        :math:`\lbrack (n_y) n_x n_z \times n_r \rbrack`.
    engine : :obj:`str`, optional
        Engine used for computations (``numpy`` or ``numba``).
    dtype : :obj:`str`, optional
        Type of elements in input array.
    name : :obj:`str`, optional
        Name of operator (to be used by :func:`pylops.utils.describe.describe`)

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly (``True``) or
        not (``False``)

    Notes
    -----
    The True-amplitude Kirchhoff single-sided demigration operator synthesizes
    seismic data given a propagation velocity model :math:`v` and a source distribution
    :math:`m`.

    In forward mode:

    .. math::
        d(\mathbf{x_r}, \mathbf{x_s}, t) =
        \widetilde{w}(t) * \int_V G(\mathbf{x_r}, \mathbf{x_s}, t)
        m(\mathbf{x_s})\,\mathrm{d}\mathbf{x_s}

    where :math:`m(\mathbf{x_s})` represents the source distribution
    at every location in the subsurface, :math:`G(\mathbf{x_r}, \mathbf{x_s}, t)`
    is the Green's function from source-to-receiver, and finally :math:`\widetilde{w}(t)` is
    a filtered version of the wavelet :math:`w(t)` [1]_ (or the wavelet itself when
    ``wavfilter=False``). In our implementation, the following high-frequency
    approximation of the Green's functions is adopted:

    .. math::
        G(\mathbf{x_r}, \mathbf{x}, \omega) = a(\mathbf{x_r}, \mathbf{x})
            e^{j \omega t(\mathbf{x_r}, \mathbf{x})}

    where :math:`a(\mathbf{x_r}, \mathbf{x})` is the amplitude and
    :math:`t(\mathbf{x_r}, \mathbf{x})` is the traveltime. These must be pre-computed
    and passed directly to the operator.

    Finally, the adjoint of the demigration operator is a *migration* operator which
    projects the data in the model domain creating an image of the source distribution.

    .. [1] Safron, L. "Multicomponent least-squares Kirchhoff depth migration",
       MSc Thesis, 2018.

    """

    # Register static methods from Kirchhoff
    _identify_geometry = Kirchhoff._identify_geometry
    _traveltime_table = Kirchhoff._traveltime_table
    _wavelet_reshaping = Kirchhoff._wavelet_reshaping

    def __init__(
        self,
        z,
        x,
        t,
        recs,
        wav,
        wavcenter,
        y=None,
        wavfilter=False,
        trav=None,
        amp=None,
        engine="numpy",
        dtype="float64",
        name="K",
    ) -> None:
        # identify geometry
        (
            self.ndims,
            _,
            dims,
            self.ny,
            self.nx,
            self.nz,
            nr,
            _,
            _,
            _,
            _,
            _,
        ) = Kirchhoff._identify_geometry(z, x, recs, y=y)
        self.dt = t[1] - t[0]
        self.nt = len(t)

        # store pre-computed traveltimes and amplitudes
        self.trav = trav
        self.amp = amp

        # create wavelet operator
        if wavfilter:
            self.wav = Kirchhoff._wavelet_reshaping(
                wav, self.dt, recs.shape[0], self.ndims
            )
        else:
            self.wav = wav
        self.cop = Convolve1D(
            (nr, self.nt), h=self.wav, offset=wavcenter, axis=1, dtype=dtype
        )

        # dimensions
        self.nr = nr
        self.ni = np.prod(dims)
        dims = tuple(dims) if self.ndims == 2 else (dims[0] * dims[1], dims[2])
        dimsd = (nr, self.nt)
        super().__init__(dtype=np.dtype(dtype), dims=dims, dimsd=dimsd, name=name)

        self._register_multiplications(engine)

    @staticmethod
    def _trav_kirch_matvec(
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

    @staticmethod
    def _trav_kirch_rmatvec(
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

            self._kirch_matvec = jit(**numba_opts)(self._trav_kirch_matvec)
            self._kirch_rmatvec = jit(**numba_opts)(self._trav_kirch_rmatvec)

        else:
            if engine == "numba" and jit is None:
                logging.warning(jit_message)

            self._kirch_matvec = self._trav_kirch_matvec
            self._kirch_rmatvec = self._trav_kirch_rmatvec

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