"""Interface of the vendored 3C-O25 model (Pitarch 2026) for WISPstation spectra.

The set-up follows the WISPstation caller of 3C-O25 (LeXPLORE processing) and the upstream
time-series example: LiEs = Ls/Ed, LtEs = Lu/Ed, the time-series parameter set and weights,
air-mass type 4, the upstream default method (leastsq) and retrieved Rrs = Lu/Ed - Rg.
"""
from __future__ import annotations

import copy

import lmfit
import numpy as np

from ...types import ThreeCResult
from ..base import MODEL_GRID, finite_or_none, to_model_grid
from .model import rrs_model_3C_O25

# parameter set of the WISPstation caller, equal to build_parameters() of the upstream time-series example
DEFAULT_PARAMS = {
    "C": {"value": 5.0, "vary": True, "min": 0.1, "max": 50.0},
    "N": {"value": 1.0, "vary": True, "min": 0.01, "max": 100.0},
    "Y": {"value": 0.5, "vary": True, "min": 0.01, "max": 5.0},
    "SNAP": {"value": 0.015, "vary": True, "min": 0.005, "max": 0.03},
    "Sg": {"value": 0.015, "vary": True, "min": 0.005, "max": 0.03},
    "rho": {"value": 0.02, "vary": True, "min": 0.0, "max": 0.03},
    "rho_d": {"value": 0.0, "vary": True, "min": 0.0, "max": 10.0},
    "rho_s": {"value": 0.01, "vary": True, "min": -0.1, "max": 0.1},
    "alpha": {"value": 0.2, "vary": True, "min": 0.0, "max": 2.0},
    "beta": {"value": 0.05, "vary": True, "min": 0.0, "max": 1.0},
}
PARAM_NAMES = ("C", "N", "Y", "SNAP", "Sg", "rho", "rho_d", "rho_s", "alpha", "beta")


def build_parameters(specification):
    """lmfit parameters from a {name: {value, vary, min, max}} specification, in the caller's order."""
    parameters = lmfit.Parameters()
    for name in PARAM_NAMES:
        item = specification[name]
        parameters.add(name, value=float(item["value"]), vary=bool(item["vary"]),
                       min=float(item["min"]), max=float(item["max"]))
    return parameters


def build_weights(wavelength_nm):
    """Time-series weights: 0 at 750-775 nm, 5 above 800 nm, 1 elsewhere."""
    wavelength = np.asarray(wavelength_nm, dtype=float)
    weights = np.ones(wavelength.size, dtype=float)
    weights[(wavelength >= 750.0) & (wavelength <= 775.0)] = 0.0
    weights[wavelength > 800.0] = 5.0
    return weights


def fold_azimuth(raa):
    """Relative azimuth folded to 0-180 degrees (the G tables are symmetric)."""
    return float(abs((float(raa) + 180.0) % 360.0 - 180.0))


class O25PitarchModel:
    """3C-O25 fit of one WISPstation measurement.

    ``sza`` and ``vza`` are the solar and view zenith angles, ``raa`` the relative azimuth
    between sun and view direction in degrees (folded to 0-180), ``rh`` in percent and
    ``pressure`` in hPa. ``water_temp`` is not used. ``params`` of the result also holds the
    geometry and ancillary inputs of the fit (theta_s, theta_v, phi, am, rh, pressure).
    ``cost`` is the lmfit chi-square, the weighted sum of squared residuals plus the
    upstream penalty on negative Rg.

    Options
        am            air-mass type (default 4, as the WISPstation caller)
        method        lmfit method (default 'leastsq', the upstream default)
        params        parameter specification replacing DEFAULT_PARAMS (same layout)
        raa_fallback  relative azimuth used when ``raa`` is missing (default None: raise).
                      The WISPstation caller uses a nominal 135 degrees when the
                      installation azimuth is unknown.
        data_folder   folder with the 3C-O25 tables (default: bundled copy)
    """

    name = "o25"
    grid = MODEL_GRID

    def __init__(self, am: float = 4.0, method: str = "leastsq", params: dict | None = None,
                 raa_fallback: float | None = None, data_folder=None):
        self.am = float(am)
        self.method = str(method)
        self.param_spec = copy.deepcopy(DEFAULT_PARAMS if params is None else params)
        build_parameters(self.param_spec)  # check the specification once
        self.raa_fallback = None if raa_fallback is None else float(raa_fallback)
        self.wl = MODEL_GRID.astype(float)
        self.weights = build_weights(self.wl)
        self.core = rrs_model_3C_O25(data_folder=data_folder)

    def fit(self, wavelength, ed, ls, lu, sza, vza=None, raa=None, rh=None, pressure=None,
            water_temp=None) -> ThreeCResult:
        ed_i, ls_i, lu_i = to_model_grid(wavelength, self.wl, ed, ls, lu)
        if not all(np.all(np.isfinite(v)) for v in (ed_i, ls_i, lu_i)):
            raise ValueError("Lu, Ls or Ed contains non-finite values in 350-900 nm")
        if np.any(ed_i <= 0.0):
            raise ValueError("Ed must be positive in 350-900 nm")
        raa_v = finite_or_none(raa)
        if raa_v is None:
            raa_v = self.raa_fallback
        inputs = {"sza": finite_or_none(sza), "vza": finite_or_none(vza), "raa": raa_v,
                  "rh": finite_or_none(rh), "pressure": finite_or_none(pressure)}
        missing = [k for k, v in inputs.items() if v is None]
        if missing:
            raise ValueError("3C-O25 needs finite " + ", ".join(missing))
        geom = (inputs["sza"], inputs["vza"], fold_azimuth(inputs["raa"]))
        anc = (self.am, inputs["rh"], inputs["pressure"])
        li_es = ls_i / ed_i
        lt_es = lu_i / ed_i
        out, rrs_model, rg = self.core.fit_LtEs(
            wl=self.wl, LiEs=li_es, LtEs=lt_es, params=build_parameters(self.param_spec),
            weights=self.weights, geom=geom, anc=anc, method=self.method, verbose=False)
        params = {k: float(v) for k, v in out.params.valuesdict().items()}
        params.update({"theta_s": geom[0], "theta_v": geom[1], "phi": geom[2], "am": anc[0], "rh": anc[1],
                       "pressure": anc[2]})
        msg = getattr(out, "message", "")
        return ThreeCResult(wavelength=self.wl.copy(), rrs=lt_es - rg, rrs_model=np.asarray(rrs_model, dtype=float),
                            rrs_surface=np.asarray(rg, dtype=float), params=params,
                            cost=float(getattr(out, "chisqr", np.nan)), success=bool(getattr(out, "success", False)),
                            message=str(msg), nfev=int(getattr(out, "nfev", 0)))
