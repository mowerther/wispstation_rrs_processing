"""Plots of a run: illumination per overpass, measured against modelled Ed(560), QWIP and spectra.

The illumination plot follows the layout of Figure 2 of the paper: the clear-sky sine model of
the day (black) with its plus or minus 20 % envelope, the measurements within the matchup window
(green inside the envelope, red outside) and the other measurements of the day (orange), over the
local time. The annotation gives the MAPE over the window, the MAPE over the day and the decision.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .qc.illumination import build_sin_curve  # noqa: E402
from .qc.qwip import QWIP_COEFFICIENTS  # noqa: E402
from .stations import STATIONS  # noqa: E402
from .table import read_table  # noqa: E402

ENVELOPE = "#f2a7a7"
MODEL_LINE = "#111111"
INSIDE = "#1B8A3A"
OUTSIDE = "#C62828"
CONTEXT = "#F2A01F"
ALL_POINTS = "#A9A9A9"
GRID = "#D7D7D7"
DPI = 200


def _name(code) -> str:
    """Station name for the built-in stations, the code otherwise."""
    st = STATIONS.get(str(code))
    return st.name if st is not None else str(code)


def _local(values) -> pd.Series:
    """Naive local clock times from ISO strings with a UTC offset."""
    return pd.Series([pd.Timestamp(v).tz_localize(None) if pd.notna(v) else pd.NaT for v in values])


def _hours(ts: pd.Series) -> np.ndarray:
    return (ts.dt.hour + ts.dt.minute / 60 + ts.dt.second / 3600 + ts.dt.microsecond / 3.6e9).to_numpy(dtype=float)


def _seconds(ts) -> int:
    ts = pd.Timestamp(ts)
    return ts.hour * 3600 + ts.minute * 60 + ts.second


def _style(ax):
    ax.grid(True, color=GRID, linestyle="--", linewidth=0.5, alpha=0.55)
    ax.tick_params(labelsize=9)


def _pct(value) -> str:
    return "n/a" if value is None or not np.isfinite(float(value)) else f"{float(value):.1f}%"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_illumination(day: pd.DataFrame, daily_row: pd.Series, overpass: pd.Series, settings, path) -> Path | None:
    """Illumination plot of one overpass (``day``: the measurements of the station on that local date)."""
    shift = float(getattr(settings, "shift_min", 8.0))
    exponent = float(getattr(settings, "exponent", 1.8))
    window = float(getattr(settings, "window_min", 15.0))
    ed_max = float(daily_row.get("ed_max", np.nan))
    if pd.isna(daily_row.get("sunrise_local")) or pd.isna(daily_row.get("sunset_local")):
        return None
    sunrise, sunset = _seconds(daily_row["sunrise_local"]), _seconds(daily_row["sunset_local"])
    t_local = _local(day["time_local"])
    hours = _hours(t_local)
    measured = day["ed560"].to_numpy(dtype=float)
    model_pts = day["ed560_model"].to_numpy(dtype=float)
    ov_local = pd.Timestamp(overpass["overpass_time_utc"]).tz_convert(
        pd.Timestamp(day["time_local"].iloc[0]).tz).tz_localize(None)
    ov_hour = _hours(pd.Series([ov_local]))[0]
    in_window = np.abs(hours - ov_hour) * 60 <= window + 1e-9
    valid = np.isfinite(model_pts) & (model_pts > 0) & np.isfinite(measured)
    inside = valid & (np.abs(measured - model_pts) <= 0.2 * model_pts)

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    if np.isfinite(ed_max):
        t, ed = build_sin_curve(sunset - sunrise, ed_max, shift, exponent)
        th = (sunrise + t) / 3600.0
        ax.fill_between(th, np.maximum(0.8 * ed, 0), 1.2 * ed, color=ENVELOPE, alpha=0.35, zorder=0, lw=0)
        ax.plot(th, ed, color=MODEL_LINE, lw=1.7, zorder=1)
    ax.axvline(ov_hour, color="#555555", lw=0.8, ls=":", zorder=1)
    ax.scatter(hours[~in_window], measured[~in_window], color=CONTEXT, s=17, alpha=0.62, edgecolor="none", zorder=2)
    for mask, colour in ((in_window & inside, INSIDE), (in_window & ~inside, OUTSIDE)):
        ax.scatter(hours[mask], measured[mask], color=colour, s=34, alpha=0.95, edgecolor="black", lw=0.35, zorder=3)
    lo = max(0, int(np.floor(min(np.nanmin(hours), sunrise / 3600))))
    hi = min(24, int(np.ceil(max(np.nanmax(hours), sunset / 3600))))
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, None)
    ax.set_xlabel("Local time [h]", fontsize=10)
    ax.set_ylabel(r"E$_d$(560) [W m$^{-2}$ nm$^{-1}$]", fontsize=10)
    label = f"{overpass['platform']} " if overpass.get("platform") else ""
    ax.set_title(f"{_name(overpass['station'])} {daily_row['date_local']}, {label}{ov_local:%H:%M} local time",
                 fontsize=11, pad=5)
    _style(ax)
    lines = [f"Overpass window: {_pct(overpass.get('illum_mape_window'))}",
             f"WISPstation day: {_pct(daily_row.get('mape_day'))}",
             f"Decision: {overpass.get('illum_decision', 'n/a')}"]
    ax.text(0.965, 0.94, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
    return _save(fig, Path(path))


def plot_ed_scatter(meas: pd.DataFrame, pairs: pd.DataFrame, station: str, path) -> Path:
    """Measured against modelled Ed(560): all measurements and those in retained matchup windows."""
    m = meas[meas["station"] == station]
    x, y = m["ed560_model"].to_numpy(dtype=float), m["ed560"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    sel = np.zeros(len(m), dtype=bool)
    if len(pairs):
        p = pairs[(pairs["station"] == station) & pairs["in_window"] & pairs["illumination_ok"]]
        sel = m["time_utc"].isin(p["time_utc"]).to_numpy()
    lim = max(0.2, float(np.nanpercentile(np.r_[x[ok], y[ok]], 99.5)) * 1.06) if ok.any() else 1.0
    xx = np.linspace(0, lim, 300)
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.fill_between(xx, 0.8 * xx, 1.2 * xx, color=ENVELOPE, alpha=0.22, zorder=0, lw=0)
    ax.plot(xx, xx, color=MODEL_LINE, lw=1.3, zorder=1)
    for f in (0.8, 1.2):
        ax.plot(xx, f * xx, color="#555555", lw=0.8, ls="--", alpha=0.8, zorder=1)
    ax.scatter(x[ok], y[ok], s=8, color=ALL_POINTS, alpha=0.32, edgecolor="none", zorder=2)
    s = ok & sel
    ax.scatter(x[s], y[s], s=28, color=INSIDE, alpha=0.94, edgecolor="black", lw=0.25, zorder=3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"Theoretical E$_d$(560) [W m$^{-2}$ nm$^{-1}$]", fontsize=10)
    ax.set_ylabel(r"Measured E$_d$(560) [W m$^{-2}$ nm$^{-1}$]", fontsize=10)
    ax.set_title(_name(station), fontsize=11)
    _style(ax)
    ax.text(0.05, 0.95, f"n={int(ok.sum())}, retained={int(s.sum())}", transform=ax.transAxes, ha="left",
            va="top", fontsize=8.5, color="#333333")
    return _save(fig, Path(path))


def plot_qwip(meas: pd.DataFrame, station: str, qwip_max: float, path) -> Path:
    """QWIP diagram: NDI(492, 665) against AVW with the QWIP polynomial and its limits."""
    m = meas[meas["station"] == station]
    avw, ndi = m["avw_nm"].to_numpy(dtype=float), m["ndi"].to_numpy(dtype=float)
    ok = m["qwip_ok"].to_numpy(dtype=bool)
    a = np.arange(400, 631)
    poly = np.polyval(QWIP_COEFFICIENTS, a)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.plot(a, poly, color=MODEL_LINE, lw=1.4)
    for d in (-qwip_max, qwip_max):
        ax.plot(a, poly + d, color="#555555", lw=0.8, ls="--")
    ax.scatter(avw[ok], ndi[ok], s=16, color=INSIDE, edgecolor="none", alpha=0.8, label=f"|QWIP| <= {qwip_max:g}")
    ax.scatter(avw[~ok], ndi[~ok], s=16, color=OUTSIDE, edgecolor="none", alpha=0.8, label=f"|QWIP| > {qwip_max:g}")
    ax.set_xlim(440, 630)
    ax.set_ylim(-2.5, 2.0)
    ax.set_xlabel("AVW [nm]", fontsize=10)
    ax.set_ylabel("NDI", fontsize=10)
    ax.set_title(_name(station), fontsize=11)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    _style(ax)
    return _save(fig, Path(path))


def plot_spectra(hyper: pd.DataFrame, pairs: pd.DataFrame, overpass: pd.Series, path) -> Path | None:
    """3C Rrs of the measurements in the matchup window of one overpass."""
    p = pairs[(pairs["station"] == overpass["station"]) & (pairs["overpass_id"] == overpass["overpass_id"])
              & pairs["in_window"]]
    if p.empty:
        return None
    h = hyper[hyper["station"] == overpass["station"]].set_index("time_utc")
    cols = [c for c in hyper.columns if c.startswith("rrs_") and c[4:].isdigit()]
    wl = np.array([float(c[4:]) for c in cols])
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for _, r in p.iterrows():
        if r["time_utc"] not in h.index:
            continue
        y = h.loc[r["time_utc"], cols].to_numpy(dtype=float)
        style = dict(color=INSIDE, lw=1.6 if r["is_closest_usable"] else 1.0) if r["insitu_usable"] else \
            dict(color=OUTSIDE, lw=1.0, ls="--")
        ax.plot(wl, y, label=f"{r['delta_min']:+.1f} min, {r['reason']}", **style)
    ax.axhline(0, color="#555555", lw=0.6)
    ax.set_xlim(350, 900)
    ax.set_xlabel("Wavelength [nm]", fontsize=10)
    ax.set_ylabel(r"R$_{rs}$ [sr$^{-1}$]", fontsize=10)
    label = f"{overpass['platform']} " if overpass.get("platform") else ""
    ax.set_title(f"{_name(overpass['station'])} {label}{pd.Timestamp(overpass['overpass_time_utc']):%Y-%m-%d %H:%M} UTC",
                 fontsize=11)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    _style(ax)
    return _save(fig, Path(path))


def _plot_settings(settings) -> SimpleNamespace:
    """Window, QWIP limit and sine model parameters from the run settings."""
    if hasattr(settings, "illumination"):
        il = settings.illumination()
        return SimpleNamespace(window_min=float(settings.window_min), qwip_max=float(settings.qwip_max),
                               shift_min=il.shift_min, exponent=il.exponent)
    return settings


def plot_run(tables: dict, out_dir, settings) -> list[Path]:
    """All plots of a run from its tables (see :func:`wispstation_rrs.batch.process`)."""
    out = Path(out_dir)
    meas, daily = tables["measurements"], tables["daily"]
    pairs, summary, hyper = tables.get("pairs", pd.DataFrame()), tables.get("summary", pd.DataFrame()), tables["hyperspectral"]
    illum = _plot_settings(settings)
    files = []
    for station in meas["station"].unique():
        files.append(plot_ed_scatter(meas, pairs, station, out / f"ed560_{station}.png"))
        files.append(plot_qwip(meas, station, float(illum.qwip_max), out / f"qwip_{station}.png"))
    for _, o in summary.iterrows():
        t_ov = pd.Timestamp(o["overpass_time_utc"])
        m = meas[meas["station"] == o["station"]]
        if m.empty:
            continue
        tz = pd.Timestamp(m["time_local"].iloc[0]).tz
        date = t_ov.tz_convert(tz).date().isoformat()
        day = m[m["date_local"].astype(str) == date]
        d = daily[(daily["station"] == o["station"]) & (daily["date_local"].astype(str) == date)]
        tag = f"{o['station']}_{t_ov:%Y%m%dT%H%M}" + (f"_{o['platform']}" if o.get("platform") else "")
        if len(day) and len(d):
            files.append(plot_illumination(day, d.iloc[0], o, illum, out / f"illumination_{tag}.png"))
        if len(pairs):
            files.append(plot_spectra(hyper, pairs, o, out / f"spectra_{tag}.png"))
    return [f for f in files if f is not None]


def plot_folder(out_dir) -> list[Path]:
    """Make the plots again from the tables of an output folder."""
    out = Path(out_dir)
    run = json.loads((out / "run_settings.json").read_text(encoding="utf-8"))
    s = run["settings"]
    il = run["illumination"]
    settings = SimpleNamespace(window_min=s["window_min"], qwip_max=s["qwip_max"], shift_min=il["shift_min"],
                               exponent=il["exponent"])
    tables = {"measurements": read_table(out / "insitu_measurements.csv"),
              "daily": pd.read_csv(out / "illumination_daily.csv"),
              "hyperspectral": read_table(out / "rrs_hyperspectral.csv")}
    for name, file in (("pairs", "insitu_overpass_pairs.csv"), ("summary", "overpass_summary.csv")):
        tables[name] = read_table(out / file) if (out / file).is_file() else pd.DataFrame()
    return plot_run(tables, out / "plots", settings)
