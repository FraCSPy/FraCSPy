import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops.utils import dottest
from pylops.utils.wavelets import ricker
from pylops.waveeqprocessing.kirchhoff import Kirchhoff

PAR = {
    "ny": 3,
    "nx": 12,
    "nz": 20,
    "nt": 50,
    "dy": 3,
    "dx": 1,
    "dz": 2,
    "dt": 0.004,
    "nsy": 4,
    "nry": 3,
    "nsx": 6,
    "nrx": 2,
}

# Check if skfmm is available and by-pass tests using it otherwise. This is
# currently required for Travis as since we moved to Python3.8 it has
# stopped working
try:
    import skfmm  # noqa: F401

    skfmm_enabled = True
except ImportError:
    skfmm_enabled = False

v0 = 500
y = np.arange(PAR["ny"]) * PAR["dy"]
x = np.arange(PAR["nx"]) * PAR["dx"]
z = np.arange(PAR["nz"]) * PAR["dz"]
t = np.arange(PAR["nt"]) * PAR["dt"]

sy = np.linspace(y.min(), y.max(), PAR["nsy"])
sx = np.linspace(x.min(), x.max(), PAR["nsx"])
syy, sxx = np.meshgrid(sy, sx, indexing="ij")
s2d = np.vstack((sx, 2 * np.ones(PAR["nsx"])))
s3d = np.vstack((syy.ravel(), sxx.ravel(), 2 * np.ones(PAR["nsx"] * PAR["nsy"])))

ry = np.linspace(y.min(), y.max(), PAR["nry"])
rx = np.linspace(x.min(), x.max(), PAR["nrx"])
ryy, rxx = np.meshgrid(ry, rx, indexing="ij")
r2d = np.vstack((rx, 2 * np.ones(PAR["nrx"])))
r3d = np.vstack((ryy.ravel(), rxx.ravel(), 2 * np.ones(PAR["nrx"] * PAR["nry"])))

wav, _, wavc = ricker(t[:21], f0=40)

par1 = {"mode": "analytic", "dynamic": False}
par2 = {"mode": "eikonal", "dynamic": False}
par3 = {"mode": "byot", "dynamic": False}
par1d = {"mode": "analytic", "dynamic": True}
par2d = {"mode": "eikonal", "dynamic": True}
par3d = {"mode": "byot", "dynamic": True}


def test_identify_geometry():
    """Identify geometry, check expected outputs"""
    # 2d
    (
        ndims,
        shiftdim,
        dims,
        ny,
        nx,
        nz,
        ns,
        nr,
        dy,
        dx,
        dz,
        dsamp,
        origin,
    ) = Kirchhoff._identify_geometry(z, x, s2d, r2d)
    assert ndims == 2
    assert shiftdim == 0
    assert [1, 2] == [1, 2]
    assert list(dims) == [PAR["nx"], PAR["nz"]]
    assert ny == 1
    assert nx == PAR["nx"]
    assert nz == PAR["nz"]
    assert ns == PAR["nsx"]
    assert nr == PAR["nrx"]
    assert list(dsamp) == [dx, dz]
    assert list(origin) == [0, 0]

    # 3d
    (
        ndims,
        shiftdim,
        dims,
        ny,
        nx,
        nz,
        ns,
        nr,
        dy,
        dx,
        dz,
        dsamp,
        origin,
    ) = Kirchhoff._identify_geometry(z, x, s3d, r3d, y=y)
    assert ndims == 3
    assert shiftdim == 1
    assert list(dims) == [PAR["ny"], PAR["nx"], PAR["nz"]]
    assert ny == PAR["ny"]
    assert nx == PAR["nx"]
    assert nz == PAR["nz"]
    assert ns == PAR["nsy"] * PAR["nsx"]
    assert nr == PAR["nry"] * PAR["nrx"]
    assert list(dsamp) == [dy, dx, dz]
    assert list(origin) == [0, 0, 0]