"""Illumination criterion: measured Ed(560) against a clear sky sine model of the day.

The criterion follows the percentile and MAPE filter of the paper processing:

* Ed is taken at the band nearest 560 nm (within 5 nm).
* EdMax of a date is the 90th percentile of the Ed values measured within 15 min of solar noon
  on all dates within plus or minus 15 days (the noon samples are pooled). An EdMax table
  (:func:`read_edmax_csv`) replaces the pooled value for its dates, for example when only part of
  the archive is at hand.
* The modelled Ed of a measurement is ``EdMax * sin(pi * t / L) ** 1.8``, with ``t`` the time
  after sunrise minus an 8 min shift and ``L`` the length of the day (sunrise to sunset), zero
  outside the day.
* The MAPE over the measurements within 15 min of an overpass decides: the overpass is retained
  when the MAPE is at most 20 % and at least one measurement lies in the window.

Times are handled as in the paper processing: local clock times of the station time zone, the solar
schedule of :func:`wispstation_rrs.geometry.solar_schedule` and whole seconds after midnight for
the model. All parameters are in :class:`IlluminationSettings`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ..geometry import solar_schedule_table
from ..stations import get_station


@dataclass
class IlluminationSettings:
    """Parameters of the illumination check (defaults: the paper settings)."""

    target_wl: float = 560.0
    wl_tol: float = 5.0
    percentile: float = 90.0
    window_days: int = 15
    noon_window_min: float = 15.0
    overpass_window_min: float = 15.0
    shift_min: float = 8.0
    exponent: float = 1.8
    mape_threshold: float = 20.0
    min_points: int = 1


def _settings(settings: IlluminationSettings | None, overrides: dict) -> IlluminationSettings:
    s = settings or IlluminationSettings()
    return IlluminationSettings(**{**asdict(s), **overrides}) if overrides else s


def _station(spectra, station=None):
    if station is not None:
        return get_station(station)
    st = spectra.attrs.get("station_info") if getattr(spectra, "attrs", None) else None
    return st if st is not None else get_station(spectra.station)


def pick_wavelength_index(wavelength, target_wl: float, tol: float) -> tuple[int | None, float | None]:
    """Index and value of the band nearest ``target_wl``, or (None, None) if it is further than ``tol``."""
    wl = np.asarray(wavelength, dtype=float).ravel()
    if wl.size == 0:
        return None, None
    i = int(np.nanargmin(np.abs(wl - float(target_wl))))
    if not np.isfinite(wl[i]) or abs(wl[i] - float(target_wl)) > float(tol):
        return None, None
    return i, float(wl[i])


def calculate_mape(measured, modeled):
    measured = np.asarray(measured, dtype=float)
    modeled = np.asarray(modeled, dtype=float)
    if measured.shape != modeled.shape or measured.size == 0: return None
    m = np.isfinite(measured) & np.isfinite(modeled) & (modeled != 0)
    if not np.any(m): return None
    return float(np.mean(np.abs((measured[m] - modeled[m]) / modeled[m])) * 100.0)


def build_sin_curve(day_length_sec, ed_max, shift_minutes, exponent):
    if ed_max is None or not np.isfinite(ed_max) or day_length_sec <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    t = np.arange(0, int(day_length_sec), dtype=float)
    shift = float(shift_minutes) * 60.0
    arg = np.clip(t, 0, day_length_sec) / float(day_length_sec)
    ed = float(ed_max) * (np.sin(np.pi * arg) ** float(exponent))
    return t + shift, ed


def day_seconds(ts) -> int:
    return ts.hour * 3600 + ts.minute * 60 + ts.second


def ed_series(spectra, settings: IlluminationSettings | None = None, station=None) -> tuple[pd.DataFrame, float | None]:
    """Ed at the target band for every measurement, with local times.

    Returns (table, actual wavelength). The table has ``time_utc``, ``time_local`` (naive local
    clock time), ``date_local`` and ``ed``, sorted by time, without NaN values and without
    duplicate times. Times are truncated to microseconds as in the paper processing.
    """
    s = settings or IlluminationSettings()
    st = _station(spectra, station)
    i, wl = pick_wavelength_index(spectra.wavelength, s.target_wl, s.wl_tol)
    cols = ["time_utc", "time_local", "date_local", "ed"]
    if i is None:
        return pd.DataFrame(columns=cols), None
    t_utc = pd.to_datetime(spectra.meta["time_utc"], utc=True).dt.floor("us").reset_index(drop=True)
    df = pd.DataFrame({
        "time_utc": t_utc,
        "time_local": t_utc.dt.tz_convert(st.timezone).dt.tz_localize(None),
        "ed": np.asarray(spectra.ed[:, i], dtype=float),
    })
    df = df.dropna(subset=["time_local", "ed"])
    df = df.drop_duplicates(subset=["time_local"]).sort_values("time_local", kind="stable").reset_index(drop=True)
    df["date_local"] = df["time_local"].dt.date
    return df[cols], wl


def schedule(dates, station) -> pd.DataFrame:
    """Solar schedule of the station for local dates (see :func:`~wispstation_rrs.geometry.solar_schedule_table`)."""
    st = get_station(station)
    return solar_schedule_table(dates, st.lat, st.lon, st.timezone)


def noon_samples(ed_df: pd.DataFrame, station, settings: IlluminationSettings | None = None) -> pd.DataFrame:
    """The Ed values measured within ``noon_window_min`` of solar noon, per local date."""
    s = settings or IlluminationSettings()
    cols = ["date_local", "time_utc", "time_local", "ed", "noon_local"]
    if ed_df.empty:
        return pd.DataFrame(columns=cols)
    sch = schedule(ed_df["date_local"].unique(), station).set_index("date_local")
    noon = ed_df["date_local"].map(sch["noon_local"])
    w = timedelta(minutes=float(s.noon_window_min))
    keep = (ed_df["time_local"] >= noon - w) & (ed_df["time_local"] <= noon + w) & np.isfinite(ed_df["ed"])
    out = ed_df[keep.to_numpy(dtype=bool)].copy()
    out["noon_local"] = noon[keep.to_numpy(dtype=bool)].to_numpy()
    return out[cols].reset_index(drop=True)


def read_edmax_csv(path) -> pd.DataFrame:
    """Read an EdMax table: ``date_local`` (local date), ``ed_max`` and optional ``station``.

    The ``illumination_daily.csv`` output of a run has these columns and can be read back.
    """
    df = pd.read_csv(path, comment="#")
    missing = {"date_local", "ed_max"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: the columns {', '.join(sorted(missing))} are required")
    df["date_local"] = pd.to_datetime(df["date_local"]).dt.date
    df["ed_max"] = pd.to_numeric(df["ed_max"], errors="coerce")
    return df


def _edmax_table(table, station_code=None) -> pd.DataFrame:
    """EdMax rows of one station (rows without a station apply to every station), one per date.

    A row of the station replaces a row without a station for the same date.
    """
    df = pd.DataFrame(table).copy()
    df["date_local"] = pd.to_datetime(df["date_local"]).dt.date
    df["ed_max"] = pd.to_numeric(df["ed_max"], errors="coerce")
    if "station" in df.columns:
        code = df["station"].astype("string").str.strip().fillna("").str.upper().to_numpy(dtype=object)
    else:
        code = np.full(len(df), "", dtype=object)
    own = code == str(station_code).upper() if station_code is not None else np.zeros(len(df), dtype=bool)
    keep = np.isfinite(df["ed_max"].to_numpy(dtype=float))
    if station_code is not None:
        keep &= own | (code == "")
    df = df.assign(_own=own)[keep]
    # rows of the station come last, so they win over rows without a station
    df = df.sort_values("_own", kind="stable").drop_duplicates("date_local", keep="last")
    return df.drop(columns="_own").sort_values("date_local").reset_index(drop=True)


def edmax_by_date(ed_df: pd.DataFrame, station, dates=None, settings: IlluminationSettings | None = None,
                  table=None) -> pd.DataFrame:
    """EdMax per local date: the percentile of the noon samples pooled over plus or minus ``window_days``.

    ``dates`` default to the dates of ``ed_df``. ``table`` (see :func:`read_edmax_csv`) gives EdMax
    for its dates instead. Returns ``date_local``, ``ed_max`` (NaN without noon samples),
    ``ed_max_source`` (``noon``, ``table`` or empty), ``n_noon_samples`` and ``n_noon_days`` (dates
    contributing samples).
    """
    s = settings or IlluminationSettings()
    samples = noon_samples(ed_df, station, s)
    by_day = {d: g["ed"].to_numpy(dtype=float) for d, g in samples.groupby("date_local")}
    if dates is None:
        dates = ed_df["date_local"].unique() if not ed_df.empty else []
    given = {}
    if table is not None:
        tab = _edmax_table(table, get_station(station).code)
        given = dict(zip(tab["date_local"], tab["ed_max"].astype(float)))
    rows = []
    for day in sorted({pd.Timestamp(d).date() for d in dates}):
        vals = []
        for k in range(-int(s.window_days), int(s.window_days) + 1):
            a = by_day.get(day + timedelta(days=k))
            if a is not None and len(a):
                vals.append(a)
        x = np.concatenate(vals) if vals else np.array([], dtype=float)
        x = x[np.isfinite(x)]
        ed_max = float(np.percentile(x, float(s.percentile))) if x.size else np.nan
        source = "noon" if x.size else ""
        if day in given:
            ed_max, source = float(given[day]), "table"
        rows.append({"date_local": day, "ed_max": ed_max, "ed_max_source": source, "n_noon_samples": int(x.size),
                     "n_noon_days": len(vals)})
    return pd.DataFrame(rows, columns=["date_local", "ed_max", "ed_max_source", "n_noon_samples", "n_noon_days"])


def modelled_ed(ed_df: pd.DataFrame, station, settings: IlluminationSettings | None = None,
                edmax: pd.DataFrame | None = None) -> pd.DataFrame:
    """Modelled Ed and the relative error of every measurement.

    Adds to ``ed_df`` the columns ``ed_max``, ``sunrise_local``, ``sunset_local``, ``day_length_s``,
    ``t_since_sunrise_s``, ``ed_model``, ``rel_error_percent`` (100 * (measured - model) / model)
    and ``abs_rel_error_percent``. ``ed_model`` is NaN for dates without EdMax or solar schedule,
    and the errors are NaN where the model is zero (before sunrise and after sunset).
    """
    s = settings or IlluminationSettings()
    out = ed_df.copy().reset_index(drop=True)
    for col in ("ed_max", "day_length_s", "t_since_sunrise_s", "ed_model", "rel_error_percent",
                "abs_rel_error_percent"):
        out[col] = np.nan
    out["sunrise_local"] = pd.NaT
    out["sunset_local"] = pd.NaT
    if out.empty:
        return out
    if edmax is None:
        edmax = edmax_by_date(ed_df, station, settings=s)
    em = edmax.set_index("date_local")["ed_max"]
    sch = schedule(out["date_local"].unique(), station).set_index("date_local")
    for day, idx in out.groupby("date_local").groups.items():
        row = sch.loc[day]
        if pd.isna(row["sunrise_local"]) or pd.isna(row["sunset_local"]):
            continue
        t_sunrise_sec = day_seconds(row["sunrise_local"])
        t_sunset_sec = day_seconds(row["sunset_local"])
        day_len = t_sunset_sec - t_sunrise_sec
        out.loc[idx, "sunrise_local"] = row["sunrise_local"]
        out.loc[idx, "sunset_local"] = row["sunset_local"]
        out.loc[idx, "day_length_s"] = day_len
        if day_len <= 0:
            continue
        sec = out.loc[idx, "time_local"].map(day_seconds).to_numpy(dtype=float)
        out.loc[idx, "t_since_sunrise_s"] = sec - t_sunrise_sec
        ed_max = em.get(day, np.nan)
        ed_max = None if pd.isna(ed_max) else float(ed_max)
        out.loc[idx, "ed_max"] = np.nan if ed_max is None else ed_max
        t_model, ed_model = build_sin_curve(day_len, ed_max, s.shift_min, s.exponent)
        if t_model.size == 0 or ed_model.size == 0:
            continue
        f = interp1d(t_model, ed_model, bounds_error=False, fill_value=0.0)
        out.loc[idx, "ed_model"] = f(sec - t_sunrise_sec)
    model = out["ed_model"].to_numpy(dtype=float)
    meas = out["ed"].to_numpy(dtype=float)
    ok = np.isfinite(model) & np.isfinite(meas) & (model != 0)
    rel = np.full(len(out), np.nan)
    rel[ok] = (meas[ok] - model[ok]) / model[ok] * 100.0
    out["rel_error_percent"] = rel
    out["abs_rel_error_percent"] = np.abs(rel)
    return out


def _decide(mape, npts: int, s: IlluminationSettings) -> tuple[str, str]:
    decision = "reject"
    reason = "MAPE_UNDEFINED" if mape is None else "MAPE_GT_THRESHOLD"
    if mape is not None and npts < int(s.min_points):
        reason = "INSUFFICIENT_POINTS"
    elif mape is not None and npts >= int(s.min_points) and mape <= float(s.mape_threshold):
        decision = "retain"
        reason = "OK"
    return decision, reason


def daily_metrics(model_df: pd.DataFrame, edmax: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per local date: EdMax, MAPE of the whole day, sunrise, sunset and number of measurements.

    ``edmax`` (the output of :func:`edmax_by_date`) adds ``ed_max_source`` and ``n_noon_samples``.
    """
    rows = []
    for day, g in model_df.groupby("date_local"):
        has_model = bool(np.isfinite(g["ed_model"].to_numpy(dtype=float)).any())
        rows.append({
            "date_local": day,
            "ed_max": g["ed_max"].iloc[0],
            "mape_day": calculate_mape(g["ed"], g["ed_model"]) if has_model else None,
            "n_points_day": int(len(g)),
            "sunrise_local": g["sunrise_local"].iloc[0],
            "sunset_local": g["sunset_local"].iloc[0],
        })
    out = pd.DataFrame(rows, columns=["date_local", "ed_max", "mape_day", "n_points_day", "sunrise_local", "sunset_local"])
    if edmax is not None and len(out):
        info = edmax.set_index("date_local")
        out.insert(2, "ed_max_source", out["date_local"].map(info["ed_max_source"]).fillna(""))
        out.insert(3, "n_noon_samples", out["date_local"].map(info["n_noon_samples"]).fillna(0).astype(int))
    return out


def mape_window(model_df: pd.DataFrame, times_utc, station, settings: IlluminationSettings | None = None,
                window_min: float | None = None) -> pd.DataFrame:
    """MAPE of the measurements within ``window_min`` (default ``overpass_window_min``) of each time.

    ``times_utc`` are overpass times. Only measurements of the same local date count. Returns one
    row per time with ``time_utc``, ``time_local``, ``date_local``, ``ed_max``, ``mape_day``,
    ``n_points_window``, ``mape_window``, ``closest_time_utc``, ``decision`` and ``reason``.
    """
    s = settings or IlluminationSettings()
    w = timedelta(minutes=float(s.overpass_window_min if window_min is None else window_min))
    st = get_station(station)
    times = pd.DatetimeIndex(pd.to_datetime(list(times_utc), utc=True))
    days = {d: g for d, g in model_df.groupby("date_local")} if not model_df.empty else {}
    daily = daily_metrics(model_df).set_index("date_local") if not model_df.empty else None
    rows = []
    for t in times:
        t_local = t.tz_convert(st.timezone).tz_localize(None)
        day = t_local.date()
        row = {"time_utc": t, "time_local": t_local, "date_local": day, "ed_max": np.nan, "mape_day": None,
               "n_points_window": 0, "mape_window": None, "closest_time_utc": pd.NaT}
        g = days.get(day)
        if g is None or g.empty:
            row.update(decision="reject", reason="NO_WISP_DAY_DATA")
            rows.append(row)
            continue
        row["ed_max"] = g["ed_max"].iloc[0]
        if pd.isna(g["sunrise_local"].iloc[0]) or not (g["day_length_s"].iloc[0] > 0):
            row.update(decision="reject", reason="NO_SOLAR_SCHEDULE")
            rows.append(row)
            continue
        if not np.isfinite(g["ed_model"].to_numpy(dtype=float)).any():
            row.update(decision="reject", reason="NO_MODEL")
            rows.append(row)
            continue
        row["mape_day"] = daily.loc[day, "mape_day"]
        win = g[(g["time_local"] >= t_local - w) & (g["time_local"] <= t_local + w)]
        npts = int(len(win))
        if npts == 0:
            row.update(decision="reject", reason="NO_WISP_WINDOW_DATA")
            rows.append(row)
            continue
        dt_abs = (win["time_local"] - t_local).abs()
        row["closest_time_utc"] = win.loc[dt_abs.idxmin(), "time_utc"]
        mape = calculate_mape(win["ed"].to_numpy(), win["ed_model"].to_numpy())
        decision, reason = _decide(mape, npts, s)
        row.update(n_points_window=npts, mape_window=mape, decision=decision, reason=reason)
        rows.append(row)
    cols = ["time_utc", "time_local", "date_local", "ed_max", "mape_day", "n_points_window", "mape_window",
            "closest_time_utc", "decision", "reason"]
    return pd.DataFrame(rows, columns=cols)


def mape_centred(model_df: pd.DataFrame, settings: IlluminationSettings | None = None,
                 window_min: float | None = None) -> pd.DataFrame:
    """For measurements without overpasses: the MAPE of the window centred on each measurement.

    Returns ``model_df`` with ``n_points_window``, ``mape_window``, ``decision`` and ``reason``.
    """
    s = settings or IlluminationSettings()
    w = np.timedelta64(int(round(float(s.overpass_window_min if window_min is None else window_min) * 60e6)), "us")
    out = model_df.copy().reset_index(drop=True)
    n_pts = np.zeros(len(out), dtype=int)
    mapes = [None] * len(out)
    decisions, reasons = ["reject"] * len(out), ["NO_MODEL"] * len(out)
    for _, idx in out.groupby("date_local").groups.items():
        g = out.loc[idx]
        if not np.isfinite(g["ed_model"].to_numpy(dtype=float)).any():
            continue
        t = g["time_local"].to_numpy(dtype="datetime64[us]")
        ed = g["ed"].to_numpy(dtype=float)
        model = g["ed_model"].to_numpy(dtype=float)
        for j, pos in enumerate(idx):
            sel = (t >= t[j] - w) & (t <= t[j] + w)
            npts = int(sel.sum())
            mape = calculate_mape(ed[sel], model[sel])
            n_pts[pos] = npts
            mapes[pos] = mape
            decisions[pos], reasons[pos] = _decide(mape, npts, s)
    out["n_points_window"] = n_pts
    out["mape_window"] = pd.array(mapes, dtype="Float64").astype(float)
    out["decision"] = decisions
    out["reason"] = reasons
    return out


def check_overpasses(spectra, overpass_times, settings: IlluminationSettings | None = None, station=None,
                     edmax_table=None, **overrides) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the check for overpass times; returns (one row per overpass, daily metrics).

    ``overpass_times`` are timestamps, :class:`~wispstation_rrs.overpass.Overpass` objects or an
    overpass table with ``time_utc``. ``edmax_table`` gives EdMax for its dates (see
    :func:`read_edmax_csv`). Keyword arguments override single settings, for example
    ``mape_threshold=25``. The station is that of ``spectra`` unless ``station`` is given.
    """
    s = _settings(settings, overrides)
    st = _station(spectra, station)
    ed_df, actual_wl = ed_series(spectra, s, st)
    edmax = edmax_by_date(ed_df, st, settings=s, table=edmax_table)
    model = modelled_ed(ed_df, st, s, edmax=edmax)
    if isinstance(overpass_times, pd.DataFrame):
        info = overpass_times.reset_index(drop=True)
        times = info["time_utc"]
    else:
        items = list(overpass_times)
        times = [getattr(o, "time_utc", o) for o in items]
        info = pd.DataFrame({"product": [getattr(o, "product", "") for o in items],
                             "platform": [getattr(o, "platform", "") for o in items]})
    res = mape_window(model, times, st, s)
    for col in ("product", "platform"):
        if col in info.columns:
            res.insert(0, col, info[col].to_numpy())
    res.insert(0, "station", st.code)
    res["actual_wl"] = actual_wl
    daily = daily_metrics(model, edmax)
    daily.insert(0, "station", st.code)
    daily["actual_wl"] = actual_wl
    return res, daily


def check_measurements(spectra, settings: IlluminationSettings | None = None, station=None,
                       edmax_table=None, **overrides) -> pd.DataFrame:
    """Run the check for every measurement with the window centred on it (no overpasses needed)."""
    s = _settings(settings, overrides)
    st = _station(spectra, station)
    ed_df, actual_wl = ed_series(spectra, s, st)
    edmax = edmax_by_date(ed_df, st, settings=s, table=edmax_table)
    out = mape_centred(modelled_ed(ed_df, st, s, edmax=edmax), s)
    out.insert(0, "station", st.code)
    out["actual_wl"] = actual_wl
    return out
