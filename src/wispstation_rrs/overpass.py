"""Satellite overpasses from product names or time tables, and their pairing with measurements.

Sentinel-2 MSI names (``S2A_MSIL1C_20220618T102559_N0510_R108_T32TMT_...``, also ``MSIL2A``) give
the sensing start time in UTC. Sentinel-3 OLCI names (``S3A_OL_1_EFR____<start>_<stop>_...``,
also ``OL_1_ERR`` and ``OL_2_WFR``) give the start and stop of the acquisition: the overpass time
is their midpoint.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

S2_RE = re.compile(r"^(S2[ABC])_MSI(L1C|L2A)_(\d{8}T\d{6})_(?:N(\d{4})_)?(?:R(\d{3})_)?(?:T(\w{5})_)?")
S3_RE = re.compile(r"^(S3[ABC])_OL_(1_E[FR]R|2_W[FR]R)_+(\d{8}T\d{6})_(\d{8}T\d{6})_")
_EXTENSIONS = (".safe.zip", ".sen3.zip", ".safe", ".sen3", ".zip", ".nc")


@dataclass(frozen=True)
class Overpass:
    """One satellite overpass. ``time_utc`` is a timezone-aware UTC timestamp."""

    product: str
    mission: str
    platform: str
    sensor: str
    time_utc: pd.Timestamp
    tile: str | None = None
    label: str | None = None


def _strip_extension(name: str) -> str:
    name = Path(str(name).strip()).name
    low = name.lower()
    for ext in _EXTENSIONS:
        if low.endswith(ext):
            return name[: -len(ext)]
    return name


def _utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(text, format="%Y%m%dT%H%M%S")).tz_localize("UTC")


def parse_product_name(name: str) -> Overpass:
    """Parse a Sentinel-2 MSI or Sentinel-3 OLCI product name (with or without extension)."""
    product = _strip_extension(name)
    m = S2_RE.match(product)
    if m:
        return Overpass(product, "Sentinel-2", m.group(1), "MSI", _utc(m.group(3)),
                        tile=f"T{m.group(6)}" if m.group(6) else None)
    m = S3_RE.match(product)
    if m:
        start, stop = _utc(m.group(3)), _utc(m.group(4))
        return Overpass(product, "Sentinel-3", m.group(1), "OLCI", start + (stop - start) / 2)
    raise ValueError(f"Not a Sentinel-2 MSI or Sentinel-3 OLCI product name: {name!r}")


def is_product_name(name: str) -> bool:
    product = _strip_extension(name)
    return bool(S2_RE.match(product) or S3_RE.match(product))


def _read_name_lines(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line)
    return names


def read_products(path_or_list) -> list[Overpass]:
    """Overpasses from a text file (one name per line, ``#`` starts a comment line), a folder
    (entries ending in .SAFE, .SEN3, .zip or bare product names) or a list of names.

    Names in a text file or a list must all parse. Folder entries that are not product names are
    ignored. The result is sorted by time.
    """
    if isinstance(path_or_list, (list, tuple)):
        names = [str(n) for n in path_or_list]
    else:
        path = Path(path_or_list)
        if path.is_dir():
            names = [p.name for p in sorted(path.iterdir()) if is_product_name(p.name)]
        elif path.is_file():
            names = _read_name_lines(path)
        else:
            raise FileNotFoundError(f"No such file or folder: {path}")
    out = [parse_product_name(n) for n in names]
    return sorted(out, key=lambda o: (o.time_utc, o.product))


def _platform_info(platform) -> tuple[str, str, str]:
    """Normalise a platform label (for example ``sentinel-2a`` or ``S2A``) to (platform, mission, sensor)."""
    if platform is None or (isinstance(platform, float) and np.isnan(platform)):
        return "", "", ""
    text = str(platform).strip().upper().replace("SENTINEL-", "S").replace("SENTINEL", "S")
    m = re.match(r"^S([23])([ABC])$", text)
    if not m:
        return str(platform).strip(), "", ""
    mission = f"Sentinel-{m.group(1)}"
    return f"S{m.group(1)}{m.group(2)}", mission, "MSI" if m.group(1) == "2" else "OLCI"


def read_times_csv(path) -> list[Overpass]:
    """Overpasses from a CSV file with ``time_utc`` and the optional columns ``platform`` and ``label``."""
    df = pd.read_csv(path, comment="#")
    if "time_utc" not in df.columns:
        raise ValueError(f"{path}: the column time_utc is required")
    times = pd.to_datetime(df["time_utc"], utc=True, format="ISO8601")
    out = []
    for i, t in enumerate(times):
        platform, mission, sensor = _platform_info(df["platform"].iloc[i] if "platform" in df.columns else None)
        label = df["label"].iloc[i] if "label" in df.columns else None
        label = None if label is None or (isinstance(label, float) and np.isnan(label)) else str(label)
        out.append(Overpass(label or "", mission, platform, sensor, pd.Timestamp(t), label=label))
    return sorted(out, key=lambda o: (o.time_utc, o.product))


def overpasses_frame(overpasses) -> pd.DataFrame:
    """Overpasses as a DataFrame (accepts a list of :class:`Overpass`, timestamps or a DataFrame)."""
    if isinstance(overpasses, pd.DataFrame):
        df = overpasses.copy()
        if "time_utc" not in df.columns:
            raise ValueError("the overpass table needs a time_utc column")
    else:
        rows = []
        for o in overpasses:
            if isinstance(o, Overpass):
                rows.append(asdict(o))
            else:
                rows.append({"product": "", "mission": "", "platform": "", "sensor": "",
                             "time_utc": pd.Timestamp(o), "tile": None, "label": None})
        df = pd.DataFrame(rows, columns=["product", "mission", "platform", "sensor", "time_utc", "tile", "label"])
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True).dt.as_unit("ns")
    for col in ("product", "platform", "sensor", "mission"):
        if col not in df.columns:
            df[col] = ""
    return df.reset_index(drop=True)


def unique_overpasses(overpasses) -> pd.DataFrame:
    """One row per platform and time: products of the same acquisition (several tiles) are joined."""
    df = overpasses_frame(overpasses)
    if df.empty:
        return df
    df["product"] = df["product"].fillna("").astype(str)
    agg = (df.groupby(["platform", "time_utc"], sort=False, dropna=False)
             .agg(product=("product", lambda s: ";".join(dict.fromkeys(x for x in s if x))),
                  mission=("mission", "first"), sensor=("sensor", "first"))
             .reset_index())
    return agg.sort_values("time_utc", kind="stable").reset_index(drop=True)[
        ["product", "mission", "platform", "sensor", "time_utc"]]


def pair(spectra_meta, overpasses, window_min: float = 15.0) -> pd.DataFrame:
    """Pair measurements with overpasses.

    ``spectra_meta`` is the ``meta`` table of a :class:`~wispstation_rrs.types.Spectra` (or the
    Spectra itself). Products of the same acquisition are counted as one overpass. Returns one
    row per overpass and measurement with ``|delta| <= window_min`` minutes: ``overpass_id``,
    ``product``, ``platform``, ``sensor``, ``overpass_time_utc``, ``station``,
    ``measurement_index`` (row of ``meta``), ``time_utc``, ``delta_min`` (measurement minus
    overpass), ``abs_delta_min`` and ``is_closest`` (the measurement nearest to the overpass).
    """
    meta = spectra_meta.meta if hasattr(spectra_meta, "meta") else spectra_meta
    ov = unique_overpasses(overpasses)
    t_meas = pd.DatetimeIndex(pd.to_datetime(meta["time_utc"], utc=True)).as_unit("ns")
    t_ns = t_meas.asi8
    station = meta["station"].to_numpy() if "station" in meta.columns else np.full(len(meta), "")
    rows = []
    for k, o in ov.iterrows():
        t0 = pd.Timestamp(o["time_utc"]).as_unit("ns").value
        delta_min = (t_ns - t0) / 6e10
        idx = np.flatnonzero(np.abs(delta_min) <= float(window_min))
        if idx.size == 0:
            continue
        closest = idx[np.argmin(np.abs(delta_min[idx]))]
        for i in idx:
            rows.append({
                "overpass_id": k, "product": o["product"], "platform": o["platform"], "sensor": o["sensor"],
                "overpass_time_utc": o["time_utc"], "station": station[i], "measurement_index": int(i),
                "time_utc": t_meas[i], "delta_min": float(delta_min[i]), "abs_delta_min": float(abs(delta_min[i])),
                "is_closest": bool(i == closest),
            })
    cols = ["overpass_id", "product", "platform", "sensor", "overpass_time_utc", "station", "measurement_index",
            "time_utc", "delta_min", "abs_delta_min", "is_closest"]
    return pd.DataFrame(rows, columns=cols)
