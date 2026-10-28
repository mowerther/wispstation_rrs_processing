"""Regression of the JAX 3C implementation against the archived Rrs of the paper processing.

The references in tests/data/threec_reference_*.csv hold, for four L1 times of each example
day, the inputs of the paper processing (sza, theta_view, rh, pressure, water temperature) and
the archived Rrs. Ed, Ls and Lu are read from the example L1 files.

Tolerance (maximum absolute Rrs difference, sr-1)
    Greifensee 1e-6: the archive stores small values with three significant digits
    (half unit up to 5e-8) and one fit (10:00 UTC) sits in a flat cost valley.
    Lake Geneva and Trasimeno 1e-9: full-precision archive.
Achieved in the repo environment (scipy 1.17, lmfit 1.3.4): Greifensee 1.4e-7,
Lake Geneva 2.0e-11, Trasimeno 1.2e-12.
"""
import hashlib
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest

from wispstation_rrs.threec import get_model
from wispstation_rrs.threec.jax_werther.model import PARAM_ORDER, WL, paper_params, paper_weights

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
L1 = HERE.parent / "examples" / "l1"

CASES = [
    ("greifensee", "20220618", "L1B_Lake_Greifen_Sprectral_20220618_040005.nc", 1e-6),
    ("geneva", "20250528", "L1_Lexplore_Sprectral_20250528_060005.nc", 1e-9),
    ("trasimeno012", "20250710", "WISPstation012_20250710_L1_SPEC.nc", 1e-9),
]

# content digests of the spectra as loaded by the paper processing (a_w, daw_dT, astar_ph, astar_y)
PAPER_SPECTRA_DIGESTS = {
    "a_w": "fd5d1b08c9f72cc032887a26ec89ce1d537f0c85027d0af7ba44f4c5d5a8eeef",
    "daw_dT": "ad6dc11c1c234c4418ced2a62db608f48281a776ef5cac5e6b33b244fa9c6e59",
    "astar_ph": "e45e2e58613f7609e58c994b71fceb12cc792754418ec592de136a3a78ef8ef8",
    "astar_y": "8673adf9a08816ff2825697044b7f273805dcf9a9afb7a3b2b04224af6e97dca",
}

# parameter table of the paper processing: name, value, vary, min, max
PAPER_TABLE = [
    ("beta", 0.5, True, 0.05, 7.0), ("alpha", 1.2, True, 0.1, 2.5), ("am", 1.0, False, 1.0, 10.0),
    ("rh", 60.0, False, 0.0, 100.0), ("pressure", 1013.25, False, 950.0, 1050.0),
    ("C_chl", 5.0, True, 0.01, 100.0), ("C_sm", 1.0, True, 0.01, 100.0), ("C_mie", 0.0, False, 0.0, 100.0),
    ("n_mie", -1.0, False, -2.0, 2.0), ("C_y", 0.5, True, 0.01, 5.0), ("S_y", 0.018, False, 0.01, 0.03),
    ("T_w", 20.0, False, 0.0, 35.0), ("theta_sun", 40.0, False, 0.0, 90.0),
    ("theta_view", 40.0, False, 0.0, 180.0), ("n_w", 1.34, False, 1.33, 1.34),
    ("rho_s", 0.0256, False, 0.0, 0.1), ("rho_dd", 0.0, True, 0.0, 0.1), ("rho_ds", 0.01, True, 0.0, 0.1),
    ("delta", 0.0, False, 0.0, 1.0),
]


def read_l1(name, index):
    """Wavelength, Ed, Ls (sky), Lu of one measurement; dims (wavelength, time) in all three files."""
    with netCDF4.Dataset(L1 / name) as ds:
        wl = np.asarray(ds["wavelength"][:], dtype=float)
        ed, ls, lu = (np.asarray(ds[v][:], dtype=float)[:, index] for v in ("downirr", "downrad", "uprad"))
    return wl, ed, ls, lu


def read_reference(station, date):
    return pd.read_csv(DATA / f"threec_reference_{station}_{date}.csv", dtype=str, keep_default_na=False)


def optional(text):
    return float(text) if text.strip() not in ("", "nan", "NaN") else None


def array_digest(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def model():
    return get_model("jax")


def test_spectra_equal_paper(model):
    for name, digest in PAPER_SPECTRA_DIGESTS.items():
        assert array_digest(getattr(model.core, name)) == digest, name
    assert np.array_equal(model.core.wl, WL)
    assert np.count_nonzero(model.core.daw_dT) > 500


def test_parameter_table_and_weights():
    p = paper_params()
    assert list(p.keys()) == PARAM_ORDER
    table = [(n, float(p[n].value), bool(p[n].vary), float(p[n].min), float(p[n].max)) for n in p]
    assert table == PAPER_TABLE
    w = pd.Series(paper_weights(WL), index=WL)
    assert (w.loc[350:500] == 5).all() and w.loc[501] == 1
    assert (w.loc[675:750] == 0.1).all() and w.loc[674] == 1 and w.loc[751] == 1
    assert (w.loc[760:770] == 0.1).all() and w.loc[759] == 1 and w.loc[771] == 1
    assert w.loc[771:900].eq(1).all()


@pytest.mark.parametrize("station,date,l1,tol", CASES)
def test_rrs_matches_paper(model, station, date, l1, tol):
    ref = read_reference(station, date)
    rrs_cols = [f"Rrs_{w}" for w in WL]
    for _, r in ref.iterrows():
        wl, ed, ls, lu = read_l1(l1, int(r["l1_index"]))
        tw = optional(r["water_temp"])
        res = model.fit(wl, ed, ls, lu, float(r["sza"]), float(r["theta_view"]), None, float(r["rh"]),
                        float(r["pressure"]), tw)
        archived = np.array([float(r[c]) for c in rrs_cols])
        assert res.success, res.message
        assert res.nfev > 0 and np.isfinite(res.cost)
        assert np.array_equal(res.wavelength, WL.astype(float))
        dmax = float(np.max(np.abs(res.rrs - archived)))
        assert dmax <= tol, f"{station} {r['time_utc']}: max |dRrs| {dmax:.3e} > {tol:.0e}"
        assert res.params["theta_sun"] == float(r["sza"])
        assert res.params["T_w"] == (20.0 if tw is None else min(max(tw, 0.0), 35.0))
        np.testing.assert_allclose(res.rrs, lu / ed - res.rrs_surface, rtol=0, atol=1e-15)


def test_view_angle_clipped_and_grid_handling(model):
    ref = read_reference("geneva", "20250528").iloc[0]
    wl, ed, ls, lu = read_l1("L1_Lexplore_Sprectral_20250528_060005.nc", int(ref["l1_index"]))
    args = dict(sza=float(ref["sza"]), rh=float(ref["rh"]), pressure=float(ref["pressure"]))
    # a view angle above 180 is clipped by lmfit to the bound 180
    res = model.fit(wl, ed, ls, lu, vza=315.0, **args)
    assert res.params["theta_view"] == 180.0
    # a decreasing input grid gives the same result
    res_rev = model.fit(wl[::-1], ed[::-1], ls[::-1], lu[::-1], vza=315.0, **args)
    assert np.array_equal(res.rrs, res_rev.rrs)
    with pytest.raises(ValueError):
        model.fit(wl[10:], ed[10:], ls[10:], lu[10:], vza=45.0, **args)


@pytest.mark.parametrize("station,date,l1,tol", CASES)
def test_compiled_path_equals_paper_path(station, date, l1, tol):
    """Cost and gradient compiled once give bitwise the result of the per-fit compilation."""
    r = read_reference(station, date).iloc[1]
    wl, ed, ls, lu = read_l1(l1, int(r["l1_index"]))
    args = (wl, ed, ls, lu, float(r["sza"]), float(r["theta_view"]), None, float(r["rh"]),
            float(r["pressure"]), optional(r["water_temp"]))
    once = get_model("jax").fit(*args)
    every = get_model("jax", compile_once=False).fit(*args)
    assert np.array_equal(once.rrs, every.rrs) and np.array_equal(once.rrs_model, every.rrs_model)
    assert once.params == every.params and once.nfev == every.nfev and once.cost == every.cost
