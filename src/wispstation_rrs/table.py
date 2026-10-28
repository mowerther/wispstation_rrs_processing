"""Output tables: check columns, the usability of the in situ side and CSV writing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# check column -> reason code when the check fails, in the order of the processing
MEASUREMENT_CHECKS = {
    "qual_ok": "L1_FLAG",
    "spectra_finite": "L1_NOT_FINITE",
    "spectra_positive": "L1_NOT_POSITIVE",
    "sza_ok": "SZA",
    "threec_ok": "THREEC",
    "qwip_ok": "QWIP",
}
ILLUMINATION_CHECK = {"illumination_ok": "ILLUMINATION"}


def add_usable(df: pd.DataFrame, checks: dict) -> pd.DataFrame:
    """Add ``insitu_usable`` (all checks pass) and ``reason`` (failed checks joined by '+', or OK)."""
    ok = np.ones(len(df), dtype=bool)
    codes = [[] for _ in range(len(df))]
    for col, code in checks.items():
        passed = df[col].fillna(False).to_numpy(dtype=bool)
        ok &= passed
        for i in np.flatnonzero(~passed):
            codes[i].append(code)
    df["insitu_usable"] = ok
    df["reason"] = ["+".join(c) if c else "OK" for c in codes]
    return df


def band_columns(columns, prefix: str = "rrs_") -> list[str]:
    """Band Rrs columns (``rrs_<SENSOR>_<BAND>``) in table order."""
    return [c for c in columns if c.startswith(prefix) and c.count("_") >= 2 and not c[len(prefix):].isdigit()]


def platform_bands(row: pd.Series, platform: str) -> dict:
    """Band values of one platform from a measurement row, as ``rrs_<BAND>``."""
    prefix = f"rrs_{platform}_"
    return {"rrs_" + c[len(prefix):]: row[c] for c in row.index if c.startswith(prefix)}


def _format_time(series: pd.Series) -> pd.Series:
    out = series.astype(object).where(series.notna(), None)
    return out.map(lambda t: None if t is None else pd.Timestamp(t).isoformat())


def write_csv(df: pd.DataFrame, path) -> Path:
    """Write a table with ISO 8601 times (UTC offset kept) and floats with 12 significant digits."""
    path = Path(path)
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[col] = _format_time(s)
    out.to_csv(path, index=False, float_format="%.12g", na_rep="")
    return path


def read_table(path, time_columns=("time_utc", "overpass_time_utc", "closest_time_utc", "usable_time_utc")) -> pd.DataFrame:
    """Read an output table back with its UTC time columns parsed."""
    df = pd.read_csv(path, low_memory=False)
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    return df
