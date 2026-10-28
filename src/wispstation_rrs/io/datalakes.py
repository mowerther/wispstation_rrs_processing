"""Reader for the Datalakes L1 NetCDF files (Greifensee and Lake Geneva).

The values are read as in the paper processing: no scaling, Ed from ``downirr``,
Ls from ``downrad`` and Lu from ``uprad``. Masked values, which do not occur in the archive,
become NaN. The per-wavelength ``*_qual`` masks (0 = nothing to report) are reduced to one flag
per measurement: True when the mask is 0 at every wavelength.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

# Variable names of the radiometry and of the quality masks.
RADIOMETRY = {"ed": "downirr", "ls": "downrad", "lu": "uprad"}
QUALITY = {"ed": "downirr_qual", "ls": "downrad_qual", "lu": "uprad_qual", "rsr": "rsr_qual"}
SENSOR = {"ed": "ed", "ls": "ld", "lu": "lu"}


def get_sensor_direction(units_attr: str) -> dict:
    """Map unit attributes to angular directions based on sensor type (as in the paper processing)."""
    if 'EdF' in units_attr:  # Ed sensor
        return {0: 315, 1: 45}  # EdF=315, EdA=45
    elif 'LuS' in units_attr:  # Lu sensor
        return {0: 45, 1: 315}  # LuS=45, LuP=315
    elif 'LdS' in units_attr:  # Ld sensor
        return {0: 45, 1: 315}  # LdS=45, LdP=315
    return {}


def time_last(var, n_time: int, n_wl: int) -> tuple[np.ndarray, int]:
    """Return a (time, wavelength) float array with masked values as NaN, and their number."""
    data = np.ma.asarray(var[:], dtype=float)
    n_masked = int(np.ma.count_masked(data))
    arr = np.ma.filled(data, np.nan)
    dims = [d.lower() for d in var.dimensions]
    if len(dims) == 2 and dims[0].startswith("time") and dims[1].startswith("wavelength"):
        pass
    elif len(dims) == 2 and dims[0].startswith("wavelength"):
        arr = arr.T
    elif arr.shape == (n_wl, n_time) and n_wl != n_time:
        arr = arr.T
    if arr.shape != (n_time, n_wl):
        raise ValueError(f"{var.name} has shape {arr.shape}, expected {(n_time, n_wl)}")
    return arr, n_masked


def read_datalakes_file(path) -> dict:
    """Read one Datalakes L1 file.

    Returns a dict with ``time`` (seconds since 1970, float), ``wavelength``, ``ed``, ``ls``,
    ``lu`` as (time, wavelength) arrays, ``columns`` (per-measurement metadata) and
    ``n_masked``.
    """
    path = Path(path)
    with Dataset(path, "r") as f:
        time = np.asarray(np.ma.getdata(f.variables["time"][:]), dtype=float).ravel()
        wavelength = np.asarray(np.ma.getdata(f.variables["wavelength"][:]), dtype=float).ravel()
        n, m = time.size, wavelength.size
        out = {"time": time, "wavelength": wavelength, "n_masked": 0}
        for key, name in RADIOMETRY.items():
            out[key], n_masked = time_last(f.variables[name], n, m)
            out["n_masked"] += n_masked

        cols = {}
        for key, name in QUALITY.items():
            if name in f.variables:
                q, _ = time_last(f.variables[name], n, m)
                cols[f"{key}_qual_ok"] = np.all(q == 0, axis=1)
            else:
                cols[f"{key}_qual_ok"] = np.ones(n, dtype=bool)
        for key, name in SENSOR.items():
            if name in f.variables:
                cols[f"{key}_sensor"] = np.asarray(np.ma.getdata(f.variables[name][:]), dtype=float).ravel()
            else:
                cols[f"{key}_sensor"] = np.full(n, np.nan)

        # File-level view angle and per-measurement direction as in the metadata of the paper processing.
        if "lu" in f.variables:
            lu_var = f.variables["lu"]
            sensor_vals = np.ma.getdata(lu_var[:])
            sensor_units = lu_var.getncattr("units") if "units" in lu_var.ncattrs() else ""
            direction_map = get_sensor_direction(sensor_units)
            sensor_code = np.unique(sensor_vals)[0] if sensor_vals.size else np.nan
            theta_view = direction_map.get(sensor_code, 45)
            direction = np.vectorize(lambda v: direction_map.get(v, -999), otypes=[float])(sensor_vals) \
                if sensor_vals.size else np.zeros(0)
        else:
            theta_view = 45
            direction = np.full(n, -999.0)
        cols["theta_view"] = np.full(n, theta_view, dtype=float)
        cols["direction_deg"] = np.asarray(direction, dtype=float).ravel()
        out["columns"] = cols
    return out
