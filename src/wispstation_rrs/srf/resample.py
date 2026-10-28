"""Band resampling of hyperspectral Rrs with sensor spectral response functions (SRF).

The integration grid is the union of the measured wavelengths and the SRF wavelengths inside the
measured range. On that grid, the SRF and the linearly interpolated spectrum are combined with
trapezoid weights. The band value of one spectrum is

    sum(w * SRF * Rrs) / sum(w * SRF)

over the finite Rrs samples. For an SRF on the measured grid (MSI at 1 nm), the union grid is the
measured grid and the result equals the convolution of the paper processing. For narrow bands on
finer grids (OLCI), the union grid keeps the shape of the SRF. The coverage of a band is the SRF area over the measured range of the
spectrum (intervals between neighbouring finite samples) divided by the SRF area over its full
support. Bands with a coverage below ``min_coverage`` are NaN.

SRF tables live in ``data/`` as ``<family>_<sensor>.csv``, e.g. ``msi_s2a.csv`` or ``olci_s3b.csv``.
A table is either wide (``wavelength_nm`` plus one column per band) or long (``band``,
``wavelength_nm``, ``response``). Band tables (``<family>_bands.csv``) hold the band centres.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_SENSORS = ("S2A", "S2B", "S2C", "S3A", "S3B")

# numpy < 2 has only trapz
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _data_dir(data_dir) -> Path:
    return DATA_DIR if data_dir is None else Path(data_dir)


def _sensor_name(path: Path) -> str:
    """Sensor name from a file name: the part after the last underscore, in capitals."""
    return path.stem.rsplit("_", 1)[-1].upper()


def _srf_files(data_dir=None) -> dict[str, Path]:
    files = {}
    for path in sorted(_data_dir(data_dir).glob("*.csv")):
        if path.stem.lower().endswith("_bands"):
            continue
        name = _sensor_name(path)
        if name in files:
            raise ValueError(f"Two SRF files for sensor {name}: {files[name].name}, {path.name}")
        files[name] = path
    return files


def available_sensors(data_dir=None) -> list[str]:
    """Names of the sensors with an SRF table in the data folder, sorted."""
    return sorted(_srf_files(data_dir))


def _resolve(sensor, data_dir=None) -> tuple[str, Path]:
    path = Path(sensor)
    if path.suffix.lower() == ".csv":
        if not path.is_file():
            raise FileNotFoundError(f"SRF file not found: {path}")
        return _sensor_name(path), path
    files = _srf_files(data_dir)
    name = str(sensor).upper()
    if name not in files:
        raise ValueError(f"Unknown sensor {sensor!r}. Available: {', '.join(files) or 'none'}")
    return name, files[name]


def _read_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    if table.shape[1] == 1:
        table = pd.read_csv(path, sep=";")
    table = table.loc[:, [c for c in table.columns if not str(c).startswith("Unnamed")]]
    table.columns = [str(c).strip() for c in table.columns]
    if {"band", "wavelength_nm", "response"} <= set(table.columns):
        long = table[["band", "wavelength_nm", "response"]].copy()
        long["band"] = long["band"].astype(str).str.strip()
    else:
        # wide table: first column is the wavelength, the others are bands
        wl_col = "wavelength_nm" if "wavelength_nm" in table.columns else table.columns[0]
        bands = [c for c in table.columns if c != wl_col]
        if not bands:
            raise ValueError(f"No band columns in {path.name}")
        long = table.melt(id_vars=wl_col, value_vars=bands, var_name="band", value_name="response")
        long = long.rename(columns={wl_col: "wavelength_nm"})
        # ESA column names such as S2A_SR_AV_B8A become B8A
        long["band"] = long["band"].str.replace(r"^.*_SR_AV_", "", regex=True)
    long["wavelength_nm"] = pd.to_numeric(long["wavelength_nm"], errors="coerce")
    long["response"] = pd.to_numeric(long["response"], errors="coerce")
    long = long.dropna(subset=["wavelength_nm", "response"])
    long["response"] = long["response"].clip(lower=0.0)
    order = list(dict.fromkeys(long["band"]))
    parts = []
    for band in order:
        part = long[long["band"] == band]
        part = part.groupby("wavelength_nm", as_index=False, sort=True)["response"].mean()
        x, y = part["wavelength_nm"].to_numpy(float), part["response"].to_numpy(float)
        if len(x) < 2 or _trapezoid(y, x) <= 0:
            raise ValueError(f"Band {band} in {path.name} has no positive response")
        part.insert(0, "band", band)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


@lru_cache(maxsize=64)
def _load_cached(path: str, mtime_ns: int) -> pd.DataFrame:
    return _read_table(Path(path))


def load_srf(sensor, data_dir=None) -> pd.DataFrame:
    """SRF of one sensor (name such as ``"S2A"`` or path to a CSV file) as a band table.

    The table is in long format with the columns ``band``, ``wavelength_nm`` and ``response``,
    bands in file order and wavelengths increasing within each band. ``attrs["sensor"]`` holds
    the sensor name and ``attrs["file"]`` the file name.
    """
    name, path = _resolve(sensor, data_dir)
    table = _load_cached(str(path.resolve()), path.stat().st_mtime_ns).copy()
    table.attrs = {"sensor": name, "file": path.name}
    return table


def _profiles(table: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    out = []
    for band, part in table.groupby("band", sort=False):
        out.append((band, part["wavelength_nm"].to_numpy(float), part["response"].to_numpy(float)))
    return out


def band_centres(sensor, data_dir=None, nominal: bool = False) -> pd.Series:
    """Band centres in nm, indexed by band.

    ``centre_nm`` of the band table is the first moment of the SRF (ESA equivalent wavelength for
    MSI, srf_centre_wavelength for OLCI). ``nominal=True`` returns the nominal band centres.
    Sensors without a band table get the first moment computed from their SRF.
    """
    name, path = _resolve(sensor, data_dir)
    column = "nominal_centre_nm" if nominal else "centre_nm"
    for bands_file in sorted(path.parent.glob("*_bands.csv")):
        table = pd.read_csv(bands_file)
        if "platform" not in table or column not in table:
            continue
        rows = table[table["platform"].astype(str).str.upper() == name]
        if len(rows):
            series = rows.set_index("band")[column].astype(float)
            series.name = column
            return series
    if nominal:
        raise ValueError(f"No nominal band centres for sensor {name}")
    centres = {}
    for band, x, y in _profiles(load_srf(path)):
        centres[band] = float(_trapezoid(x * y, x) / _trapezoid(y, x))
    series = pd.Series(centres, name=column, dtype=float)
    series.index.name = "band"
    return series


def _trapz_weights(x: np.ndarray) -> np.ndarray:
    n = x.size
    if n == 1:
        return np.array([1.0])
    dx = np.diff(x)
    w = np.empty_like(x)
    w[1:-1] = (dx[1:] + dx[:-1]) / 2.0
    w[0] = dx[0] / 2.0
    w[-1] = dx[-1] / 2.0
    return w


def _operator(x: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Weights a with a @ r = trapezoid integral of SRF times the linearly interpolated r.

    The integral runs over the union of ``x`` and the SRF wavelengths ``xs`` inside the range of
    ``x``. Each union point spreads its weight to the two neighbouring measured samples.
    """
    u = np.union1d(x, xs[(xs > x[0]) & (xs < x[-1])])
    su = np.interp(u, xs, ys, left=0.0, right=0.0)
    su[su < 0] = 0.0
    wu = _trapz_weights(u) * su
    j = np.clip(np.searchsorted(x, u, side="right") - 1, 0, x.size - 2)
    t = (u - x[j]) / (x[j + 1] - x[j])
    a = np.zeros(x.size)
    np.add.at(a, j, wu * (1.0 - t))
    np.add.at(a, j + 1, wu * t)
    return a


def _cumulative_area(x_srf: np.ndarray, y_srf: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Integral of the piecewise-linear SRF from its first wavelength up to ``x``."""
    cum = np.concatenate(([0.0], np.cumsum(np.diff(x_srf) * (y_srf[1:] + y_srf[:-1]) / 2.0)))
    xc = np.clip(x, x_srf[0], x_srf[-1])
    i = np.clip(np.searchsorted(x_srf, xc, side="right") - 1, 0, len(x_srf) - 2)
    return cum[i] + (xc - x_srf[i]) * (y_srf[i] + np.interp(xc, x_srf, y_srf)) / 2.0


def _sensor_list(sensor) -> list:
    if isinstance(sensor, (str, Path)):
        return [sensor]
    return list(dict.fromkeys(sensor))


def resample(wavelength, rrs, sensor: str | Path | Iterable = DEFAULT_SENSORS,
             min_coverage: float = 0.99, data_dir=None) -> pd.DataFrame:
    """Resample hyperspectral Rrs to the bands of one or more sensors.

    ``wavelength`` (m,) in nm, strictly increasing. ``rrs`` (n, m) or (m,), NaN for missing samples.
    Returns one row per spectrum with the columns ``rrs_<SENSOR>_<BAND>`` and
    ``coverage_<SENSOR>_<BAND>`` for each sensor in turn. Band values with a coverage below
    ``min_coverage`` are NaN.
    """
    x = np.asarray(wavelength, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("wavelength must be a finite, strictly increasing 1-D array of length >= 2")
    index = rrs.index if isinstance(rrs, pd.DataFrame) else None
    r = np.asarray(rrs, dtype=float)
    if r.ndim == 1:
        r = r[None, :]
    if r.ndim != 2 or r.shape[1] != x.size:
        raise ValueError(f"rrs has shape {r.shape}, expected (n, {x.size})")
    if not 0.0 <= float(min_coverage) <= 1.0:
        raise ValueError("min_coverage must lie between 0 and 1")

    finite = np.isfinite(r)
    mask = finite.astype(float)
    r0 = np.where(finite, r, 0.0)
    # measured intervals: both neighbouring samples finite
    valid = (finite[:, :-1] & finite[:, 1:]).astype(float)

    columns = {}
    for item in _sensor_list(sensor):
        table = load_srf(item, data_dir)
        name = table.attrs["sensor"]
        profiles = _profiles(table)
        op = np.empty((x.size, len(profiles)))
        area = np.empty((x.size - 1, len(profiles)))
        total = np.empty(len(profiles))
        for k, (_, xs, ys) in enumerate(profiles):
            op[:, k] = _operator(x, xs, ys)
            cum = _cumulative_area(xs, ys, x)
            area[:, k] = np.diff(cum)
            total[k] = _trapezoid(ys, xs)
        num = r0 @ op
        den = mask @ op
        value = np.full(num.shape, np.nan)
        np.divide(num, den, out=value, where=den > 0)
        coverage = np.clip((valid @ area) / total, 0.0, 1.0)
        value[coverage < min_coverage - 1e-12] = np.nan
        bands = [band for band, _, _ in profiles]
        for k, band in enumerate(bands):
            columns[f"rrs_{name}_{band}"] = value[:, k]
        for k, band in enumerate(bands):
            columns[f"coverage_{name}_{band}"] = coverage[:, k]
    return pd.DataFrame(columns, index=index)
