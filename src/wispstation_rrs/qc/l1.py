"""Checks of the L1 radiometry, one flag per measurement."""
from __future__ import annotations

import numpy as np
import pandas as pd


def l1_flags(spectra, wl_min: float = 350.0, wl_max: float = 900.0, sza_max: float | None = None,
             sza=None) -> pd.DataFrame:
    """Per-measurement L1 flags (True = passes).

    * ``qual_ok``: the L1 quality flags of Ed, Ls and Lu are all OK.
    * ``spectra_finite``: Ed, Ls and Lu are finite at every band in ``wl_min``-``wl_max``.
    * ``spectra_positive``: Ed, Ls and Lu are positive at every band in that range.
    * ``duplicate_time``: the time equals that of an earlier measurement (True = duplicate).
    * ``sza_ok``: the solar zenith angle is at most ``sza_max``; all True when ``sza_max`` is None.
      ``sza`` (degrees, one per measurement) is computed for the station when not given.
    * ``l1_ok``: all of the above.
    """
    meta = spectra.meta
    n = len(meta)
    out = pd.DataFrame({"time_utc": pd.to_datetime(meta["time_utc"], utc=True)}, index=meta.index)

    qual = np.ones(n, dtype=bool)
    for col in ("ed_qual_ok", "ls_qual_ok", "lu_qual_ok"):
        if col in meta.columns:
            qual &= meta[col].fillna(False).to_numpy(dtype=bool)
    out["qual_ok"] = qual

    band = (spectra.wavelength >= wl_min) & (spectra.wavelength <= wl_max)
    finite = np.ones(n, dtype=bool)
    positive = np.ones(n, dtype=bool)
    for arr in (spectra.ed, spectra.ls, spectra.lu):
        a = np.asarray(arr, dtype=float)[:, band]
        finite &= np.isfinite(a).all(axis=1)
        with np.errstate(invalid="ignore"):
            positive &= (a > 0).all(axis=1)
    out["spectra_finite"] = finite
    out["spectra_positive"] = positive

    out["duplicate_time"] = pd.to_datetime(meta["time_utc"], utc=True).duplicated(keep="first").to_numpy()

    if sza_max is None:
        out["sza_ok"] = np.ones(n, dtype=bool)
    else:
        if sza is None:
            from ..geometry import solar_angles
            from ..stations import get_station
            st = spectra.attrs.get("station_info") if spectra.attrs else None
            st = st if st is not None else get_station(spectra.station)
            sza = solar_angles(meta["time_utc"], st.lat, st.lon)["sza"].to_numpy()
        sza = np.asarray(sza, dtype=float)
        with np.errstate(invalid="ignore"):
            out["sza_ok"] = np.isfinite(sza) & (sza <= float(sza_max))
    out["l1_ok"] = out["qual_ok"] & out["spectra_finite"] & out["spectra_positive"] & ~out["duplicate_time"] & out["sza_ok"]
    return out
