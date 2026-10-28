"""Readers for WISPstation L1 files and for meteorological tables."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from ..stations import DATALAKES, WISPCLOUD, Station, get_station, parse_l1_filename, wispcloud_station_id
from ..types import Spectra
from .datalakes import read_datalakes_file
from .meteo import meteo_for, read_meteo_csv
from .wispcloud import SkipFile, read_wispcloud_file

__all__ = ["read_l1_folder", "read_l1_file", "list_l1_files", "read_meteo_csv", "meteo_for"]

META_COLUMNS = ["station", "file", "time_utc", "ed_qual_ok", "ls_qual_ok", "lu_qual_ok", "rsr_qual_ok",
                "ed_sensor", "ls_sensor", "lu_sensor", "theta_view", "direction_deg"]


def list_l1_files(path, recursive: bool = True) -> list[Path]:
    """L1 NetCDF files in a folder (searched recursively), a single file or a list of either."""
    if isinstance(path, (list, tuple)):
        files = []
        for item in path:
            files.extend(list_l1_files(item, recursive=recursive))
        return sorted(dict.fromkeys(files))
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"No such file or folder: {path}")
    pattern = "**/*.nc" if recursive else "*.nc"
    return sorted(p for p in path.glob(pattern) if p.is_file())


def _content_format(path: Path) -> str | None:
    with Dataset(path, "r") as f:
        names = set(f.variables)
    if any(n.endswith("_qc") for n in names):
        return WISPCLOUD
    if any(n.endswith("_qual") for n in names):
        return DATALAKES
    return None


def _station_and_format(path: Path, station: Station | None) -> tuple[str | None, str | None]:
    code, fmt, _ = parse_l1_filename(path)
    if station is not None:
        code = station.code
        fmt = fmt or station.file_format
    if fmt is None:
        fmt = _content_format(path)
    if code is None and fmt == WISPCLOUD:
        code = wispcloud_station_id(path)
    return code, fmt


def _read_parts(files: list[Path], fmt: str, progress: bool) -> tuple[list, dict]:
    reader = read_wispcloud_file if fmt == WISPCLOUD else read_datalakes_file
    parts, skipped = [], {}
    iterator = files
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(files, desc="Reading L1 files")
    for fp in iterator:
        try:
            part = reader(fp)
        except SkipFile as exc:
            skipped[fp.name] = str(exc)
            continue
        part["file"] = fp.name
        parts.append(part)
    return parts, skipped


def _merge(code: str, station: Station, fmt: str, parts: list, skipped: dict) -> Spectra:
    wavelength = parts[0]["wavelength"]
    kept = []
    for part in parts:
        wl = part["wavelength"]
        if wl.shape != wavelength.shape or not np.allclose(wl, wavelength, equal_nan=True):
            skipped[part["file"]] = "wavelength mismatch"
            continue
        kept.append(part)

    time = np.concatenate([p["time"] for p in kept])
    ed = np.concatenate([p["ed"] for p in kept], axis=0)
    ls = np.concatenate([p["ls"] for p in kept], axis=0)
    lu = np.concatenate([p["lu"] for p in kept], axis=0)
    cols = {name: np.concatenate([p["columns"][name] for p in kept]) for name in kept[0]["columns"]}
    files = np.concatenate([np.full(p["time"].size, p["file"], dtype=object) for p in kept])

    valid = np.isfinite(time)
    n_invalid = int((~valid).sum())
    order = np.argsort(np.where(valid, time, np.inf), kind="stable")[: int(valid.sum())]
    time_sorted = time[order]
    keep = np.ones(order.size, dtype=bool)
    keep[1:] = time_sorted[1:] != time_sorted[:-1]
    n_dup = int((~keep).sum())
    idx = order[keep]

    meta = pd.DataFrame({
        "station": code,
        "file": files[idx],
        "time_utc": pd.to_datetime(time[idx], unit="s", utc=True).as_unit("ns"),
    })
    for name in META_COLUMNS[3:]:
        meta[name] = cols[name][idx]
    meta = meta.reset_index(drop=True)

    attrs = {
        "station_info": station,
        "file_format": fmt,
        "files": [p["file"] for p in kept],
        "skipped_files": dict(skipped),
        "n_duplicates_dropped": n_dup,
        "n_invalid_times_dropped": n_invalid,
        "n_masked_values": int(sum(p["n_masked"] for p in kept)),
    }
    return Spectra(code, np.asarray(wavelength, dtype=float), ed[idx], ls[idx], lu[idx], meta, attrs)


def read_l1_folder(path, station=None, recursive: bool = True, progress: bool = False) -> list[Spectra]:
    """Read all L1 files of a folder, a file or a list of files into one :class:`Spectra` per station.

    The station and the file format come from the file names (``L1B_Lake_Greifen_*``,
    ``L1_Lexplore_*``, both spellings ``Sprectral`` and ``Spectral``, ``WISPstationNNN_*``).
    ``station`` (a :class:`~wispstation_rrs.stations.Station` or a built-in code) assigns all
    files to that station, for example for a user-defined site. The measurements of all files are
    sorted by time, so days split over several files are merged, and exact duplicate times are
    dropped (the first occurrence in file name order is kept). ``attrs`` records the number of
    dropped duplicates, the files read and the files skipped.
    """
    station_obj = get_station(station) if station is not None else None
    groups: dict[str, dict] = {}
    unknown = []
    for fp in list_l1_files(path, recursive=recursive):
        code, fmt = _station_and_format(fp, station_obj)
        if fmt is None:
            unknown.append(fp.name)
            continue
        if code is None:
            unknown.append(fp.name)
            continue
        group = groups.setdefault(code, {"fmt": fmt, "files": []})
        if group["fmt"] != fmt:
            raise ValueError(f"Files of station {code} come in two formats ({group['fmt']} and {fmt}).")
        group["files"].append(fp)
    if unknown:
        raise ValueError("Cannot tell the station of: " + ", ".join(unknown[:5])
                         + (" ..." if len(unknown) > 5 else "")
                         + ". Pass station= (a built-in code or custom_station(...)).")

    out = []
    for code in sorted(groups):
        group = groups[code]
        if station_obj is not None:
            st = station_obj
        else:
            try:
                st = get_station(code)
            except KeyError:
                warnings.warn(f"No built-in station for {code}: geometry and the illumination check need "
                              "station=custom_station(...).")
                st = Station(code, code, np.nan, np.nan, np.nan, "UTC", file_format=group["fmt"])
        parts, skipped = _read_parts(group["files"], group["fmt"], progress)
        if not parts:
            warnings.warn(f"No readable L1 files for station {code}.")
            continue
        out.append(_merge(st.code, st, group["fmt"], parts, skipped))
    return out


def read_l1_file(path, station=None) -> Spectra:
    """Read one L1 file (see :func:`read_l1_folder`)."""
    result = read_l1_folder([path], station=station)
    if not result:
        raise ValueError(f"No measurements read from {path}")
    return result[0]
