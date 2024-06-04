import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pylops.utils import dottest
from pylops.utils.wavelets import ricker
from pyfrac.modelling.kirchhoff import Kirchhoff
from pyfrac.modelling.trueamp_kirchhoff import TAKirchhoff


PAR = {
    "ny": 3,
    "nx": 12,
    "nz": 20,
    "nt": 50,
    "dy": 3,
    "dx": 1,
    "dz": 2,
    "dt": 0.004,
    "nry": 3,
    "nrx": 2,
}

v0 = 500
y = np.arange(PAR["ny"]) * PAR["dy"]
x = np.arange(PAR["nx"]) * PAR["dx"]
z = np.arange(PAR["nz"]) * PAR["dz"]
t = np.arange(PAR["nt"]) * PAR["dt"]

ry = np.linspace(y.min(), y.max(), PAR["nry"])
rx = np.linspace(x.min(), x.max(), PAR["nrx"])
ryy, rxx = np.meshgrid(ry, rx, indexing="ij")
r2d = np.vstack((rx, 2 * np.ones(PAR["nrx"])))
r3d = np.vstack((ryy.ravel(), rxx.ravel(), 2 * np.ones(PAR["nrx"] * PAR["nry"])))

wav, _, wavc = ricker(t[:21], f0=40)

par1 = {"mode": "analytic"}
par2 = {"mode": "eikonal"}
par3 = {"mode": "byot"}


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
        nr,
        dy,
        dx,
        dz,
        dsamp,
        origin,
    ) = Kirchhoff._identify_geometry(z, x, r2d)
    assert ndims == 2
    assert shiftdim == 0
    assert [1, 2] == [1, 2]
    assert list(dims) == [PAR["nx"], PAR["nz"]]
    assert ny == 1
    assert nx == PAR["nx"]
    assert nz == PAR["nz"]
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
        nr,
        dy,
        dx,
        dz,
        dsamp,
        origin,
    ) = Kirchhoff._identify_geometry(z, x, r3d, y=y)
    assert ndims == 3
    assert shiftdim == 1
    assert list(dims) == [PAR["ny"], PAR["nx"], PAR["nz"]]
    assert ny == PAR["ny"]
    assert nx == PAR["nx"]
    assert nz == PAR["nz"]
    assert nr == PAR["nry"] * PAR["nrx"]
    assert list(dsamp) == [dy, dx, dz]
    assert list(origin) == [0, 0, 0]


@pytest.mark.parametrize("par", [(par1), (par2), (par3),])
def test_kirchhoff2d(par):
    """Dot-test for 2D Kirchhoff operator"""
    vel = v0 * np.ones((PAR["nx"], PAR["nz"]))

    if par["mode"] == "byot":
        trav = Kirchhoff._traveltime_table(
            z, x, r2d, v0, mode="analytic"
        )
        trav = trav.reshape(PAR["nx"] * PAR["nz"], PAR["nrx"])
    else:
        trav = None

    Dop = Kirchhoff(
        z,
        x,
        t,
        r2d,
        vel if par["mode"] == "eikonal" else v0,
        wav,
        wavc,
        trav=trav,
        mode=par["mode"],
    )
    assert dottest(Dop, PAR["nrx"] * PAR["nt"], PAR["nz"] * PAR["nx"], atol=1e-3)


@pytest.mark.parametrize("par", [(par1), (par2), (par3),])
def test_kirchhoff3d(par):
    """Dot-test for 3D Kirchhoff operator"""
    vel = v0 * np.ones((PAR["ny"], PAR["nx"], PAR["nz"]))

    if par["mode"] == "byot":
        trav = Kirchhoff._traveltime_table(
            z, x, r3d, v0, y=y, mode="analytic"
        )
        trav = trav.reshape(PAR["ny"] * PAR["nx"] * PAR["nz"], PAR["nry"] * PAR["nrx"])
    else:
        trav = None

    Dop = Kirchhoff(
        z,
        x,
        t,
        r3d,
        vel if par["mode"] == "eikonal" else v0,
        wav,
        wavc,
        y=y,
        trav=trav,
        mode=par["mode"],
    )
    assert dottest(Dop, PAR["nry"] * PAR["nrx"] * PAR["nt"], PAR["ny"] * PAR["nx"] * PAR["nz"], atol=1e-3)


@pytest.mark.parametrize("par", [(par3),])
def test_takirchhoff2d(par):
    """Dot-test for 2D True-amplitude Kirchhoff operator"""
    trav = Kirchhoff._traveltime_table(
        z, x, r2d, v0, mode="analytic"
    )
    trav = trav.reshape(PAR["nx"] * PAR["nz"], PAR["nrx"])
    amp = 1. / (trav + 1e-5)

    Dop = TAKirchhoff(
        z,
        x,
        t,
        r2d,
        wav,
        wavc,
        trav=trav,
        amp=amp,
    )
    assert dottest(Dop, PAR["nrx"] * PAR["nt"], PAR["nz"] * PAR["nx"], atol=1e-3)


@pytest.mark.parametrize("par", [(par3), ])
def test_takirchhoff3d(par):
    """Dot-test for 3D True-amplitude Kirchhoff operator"""
    trav = Kirchhoff._traveltime_table(
        z, x, r3d, v0, y=y, mode="analytic"
    )
    trav = trav.reshape(PAR["ny"] * PAR["nx"] * PAR["nz"], PAR["nry"] * PAR["nrx"])
    amp = 1. / (trav + 1e-5)

    Dop = TAKirchhoff(
        z,
        x,
        t,
        r3d,
        wav,
        wavc,
        y=y,
        trav=trav,
        amp=amp,
    )
    assert dottest(Dop, PAR["nry"] * PAR["nrx"] * PAR["nt"], PAR["ny"] * PAR["nx"] * PAR["nz"], atol=1e-3)
