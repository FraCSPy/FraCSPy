__all__ = ["Kirchhoff"]


import logging
import os
from typing import Optional, Tuple, Union

import numpy as np

from pylops import LinearOperator
from pylops.signalprocessing import Convolve1D
from pylops.utils import deps
from pylops.utils.decorators import reshaped
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
    r"""Kirchhoff single-sided, demigration operator.

    Kirchhoff-based demigration/migration operator for single-sided propagation (from
    subsurface to surface). Uses a high-frequency approximation of Green's function
    propagators based on ``trav``.

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
    mode : :obj:`str`, optional
        Computation mode (``analytic``, ``eikonal`` or ``byot``, see Notes for
        more details)
    wavfilter : :obj:`bool`, optional
        Apply wavelet filter (``True``) or not (``False``)
    trav : :obj:`numpy.ndarray` or :obj:`tuple`, optional
        Traveltime table of size
        :math:`\lbrack (n_y) n_x n_z \times n_r \rbrack` (to be provided if ``mode='byot'``).
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

    Raises
    ------
    NotImplementedError
        If ``mode`` is neither ``analytic``, ``eikonal``, or ``byot``

    Notes
    -----
    The Kirchhoff single-sided demigration operator synthesizes seismic data given a
    propagation velocity model :math:`v` and a source distribution :math:`m`.
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
        G(\mathbf{x_r}, \mathbf{x_s}, \omega) = e^{j \omega t(\mathbf{x_r}, \mathbf{x_s})}

    where :math:`t(\mathbf{x_r}, \mathbf{x})` is the traveltime and no amplitude term is applied

    Depending on the choice of ``mode`` the traveltime and amplitude of the Green's
    function will be also computed differently:

    * ``mode=analytic`` or ``mode=eikonal``: traveltimes are computed for every
      source-receiver pair and the Green's functions are implemented from traveltime
      look-up tables, placing the source distribution values at corresponding source-to-receiver
      time in the data.
    * ``byot``: bring your own tables. The traveltime table is provided
      directly by user using ``trav`` input parameter.

    Finally, the adjoint of the demigration operator is a *migration* operator which
    projects the data in the model domain creating an image of the source distribution.

    .. [1] Safron, L. "Multicomponent least-squares Kirchhoff depth migration",
       MSc Thesis, 2018.

    """

    def __init__(
        self,
        z,
        x,
        t,
        recs,
        vel,
        wav,
        wavcenter,
        y=None,
        mode="eikonal",
        wavfilter=False,
        trav=None,
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

        # compute traveltime
        if mode in ["analytic", "eikonal", "byot"]:
            if mode in ["analytic", "eikonal"]:
                # compute traveltime table
                self.trav_recs = Kirchhoff._traveltime_table(z, x, recs, vel, y=y, mode=mode)
            else:
                self.trav_recs = trav
        else:
            raise NotImplementedError("method must be analytic, eikonal or byot")

        # create wavelet operator
        if wavfilter:
            self.wav = self._wavelet_reshaping(
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
    def _identify_geometry(
        z: NDArray,
        x: NDArray,
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
        float,
        float,
        float,
        NDArray,
        NDArray,
    ]:
        """Identify geometry and acquisition size and sampling"""
        nr = recs.shape[1]
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
        return ndims, shiftdim, dims, ny, nx, nz, nr, dy, dx, dz, dsamp, origin

    @staticmethod
    def _traveltime_table(
        z: NDArray,
        x: NDArray,
        recs: NDArray,
        vel: Union[float, NDArray],
        y: Optional[NDArray] = None,
        mode: str = "eikonal",
    ) -> NDArray:
        r"""Traveltime table

        Compute traveltimes along the source-subsurface-receivers triplet
        in 2- or 3-dimensional media given a constant or depth- and space variable
        velocity.

        Parameters
        ----------
        z : :obj:`numpy.ndarray`
            Depth axis
        x : :obj:`numpy.ndarray`
            Spatial axis
        recs : :obj:`numpy.ndarray`
            Receivers in array of size :math:`\lbrack 2 (3) \times n_r \rbrack`
        vel : :obj:`numpy.ndarray` or :obj:`float`
            Velocity model of size :math:`\lbrack (n_y \times)\, n_x
            \times n_z \rbrack` (or constant)
        y : :obj:`numpy.ndarray`
            Additional spatial axis (for 3-dimensional problems)
        mode : :obj:`numpy.ndarray`, optional
            Computation mode (``eikonal``, ``analytic`` - only for constant velocity)

        Returns
        -------
        trav_recs : :obj:`numpy.ndarray`
            Receiver-to-subsurface traveltime table of size
            :math:`\lbrack (n_y) n_x n_z \times n_r \rbrack`

        """
        # define geometry
        (
            ndims,
            shiftdim,
            dims,
            ny,
            nx,
            nz,
            nr,
            _,
            _,
            _,
            dsamp,
            origin,
        ) = Kirchhoff._identify_geometry(z, x, recs, y=y)

        # compute traveltimes
        if mode == "analytic":
            if not isinstance(vel, (float, int)):
                raise ValueError("vel must be scalar for mode=analytical")

            # compute grid
            if ndims == 2:
                X, Z = np.meshgrid(x, z, indexing="ij")
                X, Z = X.ravel(), Z.ravel()
            else:
                Y, X, Z = np.meshgrid(y, x, z, indexing="ij")
                Y, X, Z = Y.ravel(), X.ravel(), Z.ravel()

            dist_recs2 = np.zeros((ny * nx * nz, nr))

            for irec, rec in enumerate(recs.T):
                dist_recs2[:, irec] = (X - rec[0 + shiftdim]) ** 2 + (
                    Z - rec[1 + shiftdim]
                ) ** 2
                if ndims == 3:
                    dist_recs2[:, irec] += (Y - rec[0]) ** 2
            trav_recs = np.sqrt(dist_recs2) / vel

        elif mode == "eikonal":
            if skfmm_message is None:
                trav_recs = np.zeros((ny * nx * nz, nr), dtype=np.float32)
                for irec, rec in enumerate(recs.T):
                    rec = np.round((rec - origin) / dsamp).astype(np.int32)
                    phi = np.ones_like(vel)
                    if ndims == 2:
                        phi[rec[0], rec[1]] = -1
                    else:
                        phi[rec[0], rec[1], rec[2]] = -1
                    trav_recs[:, irec] = (
                        skfmm.travel_time(phi=phi, speed=vel, dx=dsamp)
                    ).ravel()
            else:
                raise NotImplementedError(skfmm_message)
        else:
            raise NotImplementedError("method must be analytic or eikonal")

        return trav_recs

    @staticmethod
    def _wavelet_reshaping(
        wav: NDArray,
        dt: float,
        dimrec: int,
        dimv: int,
    ) -> NDArray:
        """Apply wavelet reshaping as from theory in [1]_"""
        f = np.fft.rfftfreq(len(wav), dt)
        W = np.fft.rfft(wav, n=len(wav))
        if dimv == 2:
            # 2D
            Wfilt = W * (2 * np.pi * f)
        elif dimrec == 2 and dimv == 3:
            # 2.5D
            raise NotImplementedError("2.D wavelet currently not available")
        elif dimrec == 3 and dimv == 3:
            # 3D
            Wfilt = W * (-1j * 2 * np.pi * f)
        wavfilt = np.fft.irfft(Wfilt, n=len(wav))
        return wavfilt

    @staticmethod
    def _trav_kirch_matvec(
        x: NDArray,
        y: NDArray,
        nr: int,
        nt: int,
        ni: int,
        dt: float,
        trav_recs: NDArray,
    ) -> NDArray:
        for irec in range(nr):
            trav = trav_recs[:, irec]
            itrav = (trav / dt).astype("int32")
            travd = trav / dt - itrav
            for ii in range(ni):
                itravii = itrav[ii]
                travdii = travd[ii]
                if 0 <= itravii < nt - 1:
                    y[irec, itravii] += x[ii] * (1 - travdii)
                    y[irec, itravii + 1] += x[ii] * travdii
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
    ) -> NDArray:
        for ii in prange(ni):
            trav_recsii = trav_recs[ii]
            for irec in range(nr):
                travii = trav_recsii[irec]
                itravii = int(travii / dt)
                travdii = travii / dt - itravii
                if 0 <= itravii < nt - 1:
                    y[ii] += (
                        x[irec, itravii] * (1 - travdii)
                        + x[irec, itravii + 1] * travdii
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
            self.trav_recs,
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
            self.trav_recs,
        )

        y = self._kirch_rmatvec(*inputs)
        return y