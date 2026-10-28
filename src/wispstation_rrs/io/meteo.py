"""Meteorological inputs of the 3C fit: relative humidity, air pressure and water temperature.

A meteo table is a CSV file with the columns ``time_utc``, optional ``station``, and any of
``rh_percent``, ``pressure_hpa`` and ``water_temperature_c``. Each measurement takes the values of
the nearest table row within a tolerance. Values that are missing fall back to the defaults: a
relative humidity of 60 %, a water temperature of 20 degC and the pressure from the station
altitude (standard barometric formula).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..stations import barometric_pressure_hpa, get_station

VALUE_COLUMNS = ["rh_percent", "pressure_hpa", "water_temperature_c"]


def read_meteo_csv(path) -> pd.DataFrame:
    """Read a meteo table. ``time_utc`` becomes timezone-aware UTC (naive values are UTC)."""
    df = pd.read_csv(path, comment="#")
    if "time_utc" not in df.columns:
        raise ValueError(f"{path}: the column time_utc is required")
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, format="ISO8601").dt.as_unit("ns")
    for col in VALUE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    if "station" in df.columns:
        df["station"] = df["station"].astype("string").str.strip()
    return df.sort_values("time_utc", kind="stable").reset_index(drop=True)


def _station_info(spectra):
    st = spectra.attrs.get("station_info") if spectra.attrs else None
    if st is not None:
        return st
    try:
        return get_station(spectra.station)
    except KeyError:
        return None


def meteo_for(spectra, table: pd.DataFrame | None = None, rh: float = 60.0, water_temp: float = 20.0,
              pressure: float | None = None, tolerance_min: float = 90) -> pd.DataFrame:
    """Meteo values for every measurement of ``spectra``.

    ``table`` is a meteo table (see :func:`read_meteo_csv`). Rows whose ``station`` matches the
    station of ``spectra``, or that have no station, are used. Each measurement takes the values
    of the nearest row within ``tolerance_min`` minutes and ``meteo_source`` is then ``"csv"``.
    Measurements without a row, and values missing in a row, take the defaults ``rh``,
    ``water_temp`` and ``pressure`` (None: barometric pressure at the station altitude), and
    ``meteo_source`` is ``"default"`` when no row was found.
    """
    if pressure is None:
        st = _station_info(spectra)
        altitude = st.altitude_m if st is not None and np.isfinite(st.altitude_m) else 0.0
        pressure = barometric_pressure_hpa(altitude)
    defaults = {"rh_percent": float(rh), "pressure_hpa": float(pressure), "water_temperature_c": float(water_temp)}

    times = pd.DatetimeIndex(pd.to_datetime(spectra.meta["time_utc"], utc=True)).as_unit("ns")
    out = pd.DataFrame({"time_utc": times, "station": spectra.station})
    for col in VALUE_COLUMNS:
        out[col] = np.nan
    out["meteo_source"] = "default"

    if table is not None and len(table):
        tab = table.copy()
        tab["time_utc"] = pd.to_datetime(tab["time_utc"], utc=True).dt.as_unit("ns")
        if "station" in tab.columns:
            st_col = tab["station"].astype("string").str.strip().fillna("").str.upper()
            keep = (st_col == "") | (st_col == str(spectra.station).upper())
            tab = tab[keep.to_numpy(dtype=bool)]
        for col in VALUE_COLUMNS:
            if col not in tab.columns:
                tab[col] = np.nan
        tab = tab[["time_utc"] + VALUE_COLUMNS].sort_values("time_utc", kind="stable")
        tab = tab.rename(columns={c: f"_{c}" for c in VALUE_COLUMNS})
        tab["_matched"] = True
        left = pd.DataFrame({"time_utc": times, "_pos": np.arange(len(times))}).sort_values("time_utc", kind="stable")
        merged = pd.merge_asof(left, tab, on="time_utc", direction="nearest",
                               tolerance=pd.Timedelta(minutes=float(tolerance_min)))
        merged = merged.sort_values("_pos").reset_index(drop=True)
        matched = merged["_matched"].eq(True).to_numpy()
        for col in VALUE_COLUMNS:
            out[col] = merged[f"_{col}"].to_numpy(dtype=float)
        out.loc[matched, "meteo_source"] = "csv"

    for col, value in defaults.items():
        out[col] = out[col].fillna(value)
    return out
