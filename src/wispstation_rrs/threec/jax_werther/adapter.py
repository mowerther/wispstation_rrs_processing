"""Interface of the JAX 3C implementation (configuration of the paper processing)."""
from __future__ import annotations

import numpy as np

from ...types import ThreeCResult
from ..base import finite_or_none, to_model_grid
from .model import SPECTRA_DIR, WL, paper_params, paper_weights, rrs_model_3C


class JaxWertherModel:
    """3C with the parameters, weights and optimiser settings of the paper processing.

    Per measurement, Ed, Ls and Lu are interpolated to 350-900 nm (1 nm), ``sza`` sets
    theta_sun, ``vza`` theta_view, ``rh`` and ``pressure`` the atmosphere and ``water_temp``
    T_w. A missing input, or one that is not finite, keeps the initial value (theta_sun 40,
    theta_view 40, rh 60, pressure 1013.25, T_w 20). lmfit clips values to the parameter bounds
    (theta_view 0-180, pressure 950-1050 hPa, T_w 0-35 degC), and ``params`` of the result holds
    the values used. ``raa`` is not used.

    Options
        screen_nonfinite  True (default): the ratios are 0 where Ed is not finite or zero, and
                          wavelengths with values that are not finite get weight 0.
                          False: plain ratios. Both give the same results for finite inputs
                          with Ed above 0.
        spectra_path      folder with phyto.A, water.A, Y.A and daWdT.txt (default: bundled).
        compile_once      True (default): cost and gradient are compiled once per model.
                          False: compiled for every fit, as in the paper processing.
                          Both give bitwise the same results, the first five to ten times faster.
    """

    name = "jax"
    grid = WL

    def __init__(self, screen_nonfinite: bool = True, spectra_path=None, compile_once: bool = True):
        self.screen_nonfinite = bool(screen_nonfinite)
        self.compile_once = bool(compile_once)
        self.core = rrs_model_3C(wl_range=(WL[0], WL[-1]),
                                 spectra_path=SPECTRA_DIR if spectra_path is None else spectra_path)
        if not np.array_equal(self.core.wl, WL):
            raise ValueError("the 3C spectra do not cover 350-900 nm at 1 nm")
        self.weights = paper_weights(WL)
        self.init_params = paper_params()

    def fit(self, wavelength, ed, ls, lu, sza, vza=None, raa=None, rh=None, pressure=None,
            water_temp=None) -> ThreeCResult:
        ed_i, ls_i, lu_i = to_model_grid(wavelength, WL, ed, ls, lu)
        params = self.init_params.copy()
        for pname, value in (("theta_sun", sza), ("theta_view", vza), ("rh", rh),
                             ("pressure", pressure), ("T_w", water_temp)):
            v = finite_or_none(value)
            if v is not None:
                params[pname].set(value=v)
        fit = self.core.fit_LuEd_compiled if self.compile_once else self.core.fit_LuEd
        reg_min, rrs_mod, rrs_refl, _, rrs_obs = fit(
            WL, ls_i, lu_i, ed_i, params, self.weights, verbose=False,
            screen_nonfinite=self.screen_nonfinite)
        return ThreeCResult(wavelength=WL.astype(float), rrs=np.asarray(rrs_obs, dtype=float),
                            rrs_model=np.asarray(rrs_mod, dtype=float),
                            rrs_surface=np.asarray(rrs_refl, dtype=float),
                            params=reg_min["final_params"], cost=float(reg_min["resid"]),
                            success=reg_min["success"], message=reg_min["message"], nfev=reg_min["nfev"])
