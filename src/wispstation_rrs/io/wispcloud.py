"""Reader for the WISPcloud L1 spectral NetCDF files (``WISPstationNNN_YYYYMMDD_L1_SPEC.nc``).

The values are read as in the paper processing: no
scaling, Ed from ``downirr``, Ls from ``downrad`` and Lu from ``uprad``, NaN where the file has
no value. The per-measurement ``*_qc`` flags (bit flags, 0 = no problem) give the quality columns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from .datalakes import get_sensor_direction, time_last

REQUIRED_VARS = ["time", "wavelength", "downirr", "downrad", "uprad", "rsr"]
RADIOMETRY = {"ed": "downirr", "ls": "downrad", "lu": "uprad"}
QUALITY = {"ed": "downirr_qc", "ls": "downrad_qc", "lu": "uprad_qc", "rsr": "rsr_qc"}
SENSOR = {"ed": "ed", "ls": "ld", "lu": "lu"}


class SkipFile(ValueError):
    """Raised for a file without the required variables or without measurements."""


def read_wispcloud_file(path) -> dict:
    """Read one WISPcloud L1 spectral file.

    Returns the same structure as :func:`wispstation_rrs.io.datalakes.read_datalakes_file`.
    Raises :class:`SkipFile` for files without the required variables or without times.
    """
    path = Path(path)
    with Dataset(path, "r") as f:
        missing = [v for v in REQUIRED_VARS if v not in f.variables]
        if missing:
            raise SkipFile(f"missing {missing}")
        time = np.asarray(f.variables["time"][:], dtype=float).ravel()
        if time.size == 0:
            raise SkipFile("no measurements")
        wavelength = np.asarray(f.variables["wavelength"][:], dtype=float)
        if wavelength.ndim != 1:
            wavelength = wavelength.squeeze()
        n, m = time.size, wavelength.size
        out = {"time": time, "wavelength": wavelength, "n_masked": 0}
        for key, name in RADIOMETRY.items():
            try:
                out[key], n_masked = time_last(f.variables[name], n, m)
            except ValueError as exc:
                raise SkipFile(str(exc)) from exc
            out["n_masked"] += n_masked

        cols = {}
        for key, name in QUALITY.items():
            if name in f.variables:
                qc = np.ma.filled(np.ma.asarray(f.variables[name][:]), 255).astype(int).ravel()
                cols[f"{key}_qual_ok"] = qc == 0
            else:
                cols[f"{key}_qual_ok"] = np.ones(n, dtype=bool)
        for key, name in SENSOR.items():
            if name in f.variables:
                cols[f"{key}_sensor"] = np.asarray(f.variables[name][:], dtype=float).ravel()
            else:
                cols[f"{key}_sensor"] = np.full(n, np.nan)

        # File-level view angle and per-measurement direction as in the metadata of the paper processing.
        if "lu" in f.variables:
            lu_var = f.variables["lu"]
            sensor_vals = np.asarray(lu_var[:], dtype=float).ravel()
            sensor_units = lu_var.getncattr("units") if "units" in lu_var.ncattrs() else ""
            direction_map = get_sensor_direction(sensor_units)
            if direction_map:
                mapped = np.vectorize(lambda v: direction_map.get(v, np.nan), otypes=[float])(sensor_vals)
            else:
                mapped = np.full(sensor_vals.shape, np.nan)
            deg_mask = np.isfinite(sensor_vals) & (sensor_vals >= 0) & (sensor_vals <= 360)
            mapped = np.where(np.isfinite(mapped), mapped, np.where(deg_mask, sensor_vals, np.nan))
            finite = mapped[np.isfinite(mapped)]
            if finite.size:
                uniq, counts = np.unique(finite, return_counts=True)
                theta_view = float(uniq[np.argmax(counts)])
            else:
                theta_view = 45.0
            direction = np.where(np.isfinite(mapped), mapped, -999)
        else:
            theta_view = 45.0
            direction = np.full(n, -999.0)
        cols["theta_view"] = np.full(n, theta_view, dtype=float)
        cols["direction_deg"] = np.asarray(direction, dtype=float).ravel()
        out["columns"] = cols
    return out
