"""Regression tests of the vendored 3C-O25 model.

tests/data/threec_o25_expected.* hold the output of the vendored copy, which was bitwise equal
to the upstream clone (v1.0.4) for the upstream example spectrum (run_3C.py defaults) and for
one Lake Geneva WISPstation measurement (WISPstation caller set-up). Tolerance 1e-9 sr-1 on the
spectra.
"""
import json
from pathlib import Path

import lmfit as lm
import netCDF4
import numpy as np
import pandas as pd
import pytest

from wispstation_rrs.threec import get_model
from wispstation_rrs.threec.o25_pitarch.adapter import fold_azimuth
from wispstation_rrs.threec.o25_pitarch import data as o25_data
from wispstation_rrs.threec.o25_pitarch.model import rrs_model_3C_O25

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
L1 = HERE.parent / "examples" / "l1"
# the installed data folder, in an editable and in a normal installation
PACKAGE_DATA = Path(o25_data.__file__).resolve().parent
TOL = 1e-9


@pytest.fixture(scope="module")
def expected():
    with open(DATA / "threec_o25_expected.json", encoding="utf-8") as f:
        meta = json.load(f)
    with np.load(DATA / "threec_o25_expected.npz") as z:
        arrays = {k: z[k] for k in z.files}
    return meta, arrays


def run3c_default_params():
    """Default parameters of the upstream scripts/run_3C.py."""
    params = lm.Parameters()
    params.add_many(
        ("C", 5.0, True, 0.1, 50.0, None), ("N", 1.0, True, 0.01, 100.0, None),
        ("Y", 0.5, True, 0.01, 5.0, None), ("SNAP", 0.015, True, 0.005, 0.03, None),
        ("Sg", 0.015, True, 0.005, 0.03, None), ("rho", 0.02, False, 0.0, 0.03, None),
        ("rho_d", 0.0, True, 0.0, 10.0, None), ("rho_s", 0.0, True, -0.01, 0.01, None),
        ("alpha", 0.2, True, 0.0, 2.0, None), ("beta", 0.05, True, 0.0, 1.0, None),
    )
    return params


def test_bundled_data_files():
    for name in ("G0w.txt", "G1w.txt", "G0p.txt", "G1p.txt", "abs_scat_seawater_20d_35PSU_20230922_short.txt",
                 "vars_aph_v2.npz"):
        assert (PACKAGE_DATA / name).is_file(), name
    m = rrs_model_3C_O25()
    assert Path(m.data_folder) == PACKAGE_DATA.resolve()
    assert m._G0w.shape == (10, 10, 13)


def test_upstream_example(expected):
    meta, arrays = expected
    ex = pd.read_csv(DATA / "threec_o25_example_single_spectrum.csv", skiprows=15, index_col=0)
    wl = ex.index.to_numpy(dtype=float)
    li, lt, es = (ex.iloc[:, k].to_numpy(dtype=float) for k in range(3))
    weights = np.where((wl >= 760) & (wl <= 765), 0.0, np.where(wl < 400, 1.0, np.where(wl > 800, 5.0, 1.0)))
    out, rrs_model, rg = rrs_model_3C_O25().fit_LtEs(
        wl=wl, LiEs=li / es, LtEs=lt / es, params=run3c_default_params(), weights=weights,
        geom=tuple(meta["example"]["geom"]), anc=tuple(meta["example"]["anc"]), verbose=False)
    assert out.success
    assert np.array_equal(wl, arrays["example_wl"])
    np.testing.assert_allclose(rrs_model, arrays["example_rrs_model"], rtol=0, atol=TOL)
    np.testing.assert_allclose(rg, arrays["example_rg"], rtol=0, atol=TOL)
    fitted = {k: out.params[k].value for k in meta["example"]["params"]}
    np.testing.assert_allclose(list(fitted.values()), list(meta["example"]["params"].values()), rtol=1e-6, atol=1e-12)


def test_wisp_measurement(expected):
    meta, arrays = expected
    inp = meta["wisp_geneva"]["input"]
    with netCDF4.Dataset(L1 / inp["l1_file"]) as ds:
        wl = np.asarray(ds["wavelength"][:], dtype=float)
        ed, ls, lu = (np.asarray(ds[v][:], dtype=float)[:, inp["l1_index"]] for v in ("downirr", "downrad", "uprad"))
    model = get_model("o25", raa_fallback=135.0)
    res = model.fit(wl, ed, ls, lu, inp["sza"], inp["vza"], None, inp["rh"], inp["pressure"], 12.0)
    assert res.success, res.message
    assert np.array_equal(res.wavelength, arrays["wisp_wl"])
    np.testing.assert_allclose(res.rrs, arrays["wisp_rrs"], rtol=0, atol=TOL)
    np.testing.assert_allclose(res.rrs_model, arrays["wisp_rrs_model"], rtol=0, atol=TOL)
    np.testing.assert_allclose(res.rrs_surface, arrays["wisp_rg"], rtol=0, atol=TOL)
    np.testing.assert_allclose(res.rrs, lu / ed - res.rrs_surface, rtol=0, atol=1e-15)
    assert res.params["phi"] == 135.0 and res.params["am"] == 4.0
    np.testing.assert_allclose(res.cost, meta["wisp_geneva"]["cost"], rtol=1e-6)
    # relative azimuth 225 is folded to 135 and gives the same fit
    res225 = model.fit(wl, ed, ls, lu, inp["sza"], inp["vza"], 225.0, inp["rh"], inp["pressure"])
    assert np.array_equal(res225.rrs, res.rrs)


def test_inputs_checked():
    assert fold_azimuth(-45.0) == 45.0 and fold_azimuth(190.0) == 170.0
    model = get_model("o25")
    wl = np.arange(350.0, 901.0)
    ones = np.ones_like(wl)
    with pytest.raises(ValueError, match="raa"):
        model.fit(wl, ones, 0.1 * ones, 0.01 * ones, 30.0, 40.0, None, 60.0, 1013.0)
    with pytest.raises(ValueError, match="rh"):
        model.fit(wl, ones, 0.1 * ones, 0.01 * ones, 30.0, 40.0, 135.0, float("nan"), 1013.0)
    with pytest.raises(ValueError, match="positive"):
        model.fit(wl, 0 * ones, 0.1 * ones, 0.01 * ones, 30.0, 40.0, 135.0, 60.0, 1013.0)
