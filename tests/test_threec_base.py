"""Tests of the common 3C interface and the process pool pattern."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from wispstation_rrs.threec import MODEL_GRID, available_models, fit_one, get_model, init_worker
from wispstation_rrs.threec.base import to_model_grid
from wispstation_rrs.types import ThreeCResult

L1 = Path(__file__).resolve().parent.parent / "examples" / "l1"


def geneva_args(index=12):
    with netCDF4.Dataset(L1 / "L1_Lexplore_Sprectral_20250528_060005.nc") as ds:
        wl = np.asarray(ds["wavelength"][:], dtype=float)
        ed, ls, lu = (np.asarray(ds[v][:], dtype=float)[:, index] for v in ("downirr", "downrad", "uprad"))
    return {"wavelength": wl, "ed": ed, "ls": ls, "lu": lu, "sza": 30.98396738298802, "vza": 45.0, "raa": 135.0,
            "rh": 74.4248275756836, "pressure": 1023.34, "water_temp": None}


def test_names():
    assert available_models() == ["jax", "o25"]
    with pytest.raises(ValueError):
        get_model("theano")


def test_to_model_grid():
    grid = MODEL_GRID
    wl = np.arange(340.0, 911.0, 0.5)
    a = np.sin(wl / 50.0)
    (out,) = to_model_grid(wl, grid, a)
    np.testing.assert_allclose(out, np.sin(grid / 50.0), atol=1e-4)
    (same,) = to_model_grid(grid.astype(float), grid, np.sin(grid / 50.0))
    assert np.array_equal(same, np.sin(grid / 50.0))
    with pytest.raises(ValueError, match="cover"):
        to_model_grid(np.arange(400.0, 901.0), grid, np.ones(501))
    with pytest.raises(ValueError):
        to_model_grid(wl, grid, a[:-1])


def test_worker_pattern_in_process():
    init_worker("o25", {"raa_fallback": 135.0})
    args = geneva_args()
    res = fit_one(args)
    assert isinstance(res, ThreeCResult) and res.success
    direct = get_model("o25").fit(**args)
    assert np.array_equal(res.rrs, direct.rrs)
    bad = dict(args, rh=float("nan"))
    failed = fit_one(bad)
    assert not failed.success and "rh" in failed.message
    assert failed.rrs.shape == MODEL_GRID.shape and np.isnan(failed.rrs).all()


@pytest.mark.parametrize("name", ["o25", "jax"])
def test_process_pool(name):
    args = geneva_args()
    direct = get_model(name).fit(**args)
    with ProcessPoolExecutor(max_workers=1, initializer=init_worker, initargs=(name, {})) as ex:
        results = list(ex.map(fit_one, [args, tuple(args.values())]))
    for res in results:
        assert res.success
        assert np.array_equal(res.rrs, direct.rrs)
        assert res.params == direct.params and res.nfev == direct.nfev
