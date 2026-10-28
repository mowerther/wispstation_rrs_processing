"""Batch processing of WISPstation L1 files to quality-filtered Rrs.

:func:`process` reads every L1 file of a folder, fits the 3C model to each measurement in a pool
of worker processes (with a progress bar), computes QWIP, the illumination check and the band
Rrs of the satellite sensors, pairs the measurements with the satellite overpasses and writes
the output tables, the run settings and the plots.

The 3C results of each station and day are kept in ``<out>/cache``. A second run with the same
inputs and 3C settings takes them from there, so an interrupted run continues with the days
that were not finished.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .geometry import solar_angles
from .io import meteo_for, read_l1_folder, read_meteo_csv
from .overpass import pair, read_products, read_times_csv, unique_overpasses
from .qc import IlluminationSettings, l1_flags, qwip, read_edmax_csv
from .qc.illumination import daily_metrics, ed_series, edmax_by_date, mape_centred, mape_window, modelled_ed
from .srf import available_sensors, resample
from .table import ILLUMINATION_CHECK, MEASUREMENT_CHECKS, add_usable, band_columns, platform_bands, write_csv
from .threec import MODEL_GRID, fit_one, init_worker
from .types import Spectra, ThreeCResult

TABLES = {
    "measurements": "insitu_measurements.csv",
    "pairs": "insitu_overpass_pairs.csv",
    "summary": "overpass_summary.csv",
    "hyperspectral": "rrs_hyperspectral.csv",
    "daily": "illumination_daily.csv",
}


@dataclass
class Settings:
    """Settings of a run (defaults: the paper processing).

    threec          3C implementation, ``jax`` (Werther) or ``o25`` (Pitarch)
    threec_options  keyword options of the implementation (see ``wispstation_rrs.threec``)
    window_min      matchup window around an overpass (minutes)
    context_min     measurements within this time of an overpass are listed in the pairs table
    rh, water_temp, pressure
                    defaults for measurements without a meteo row (pressure None: from the
                    station altitude)
    sensors         sensors of the band Rrs (None: every sensor with a band response file)
    min_coverage    minimum share of the band response covered by the spectrum
    qwip_max        limit of the absolute QWIP score
    mape_max        limit of the illumination MAPE (%)
    sza_max         limit of the solar zenith angle (None: no limit)
    fit             ``all`` measurements or only those in a matchup ``window``
    workers         number of worker processes of the 3C fits
    resume          take the 3C results of earlier runs from the cache
    """

    threec: str = "jax"
    threec_options: dict = field(default_factory=dict)
    window_min: float = 15.0
    context_min: float = 60.0
    rh: float = 60.0
    water_temp: float = 20.0
    pressure: float | None = None
    meteo_tolerance_min: float = 90.0
    sensors: tuple | None = None
    min_coverage: float = 0.99
    qwip_max: float = 0.2
    mape_max: float = 20.0
    sza_max: float | None = None
    fit: str = "all"
    workers: int = 2
    resume: bool = True
    plots: bool = True
    progress: bool = True

    def illumination(self) -> IlluminationSettings:
        return IlluminationSettings(overpass_window_min=float(self.window_min), mape_threshold=float(self.mape_max))


def load_overpasses(products=None, times=None) -> pd.DataFrame:
    """One row per acquisition from product names (file, folder or list) and a times CSV."""
    items = []
    for source in ([] if products is None else (products if isinstance(products, (list, tuple)) else [products])):
        p = Path(str(source))
        items += read_products(p) if (p.is_file() or p.is_dir()) else read_products([str(source)])
    if times is not None:
        items += read_times_csv(times)
    return unique_overpasses(items)


# 3C fits ------------------------------------------------------------------------------------------

def _task_key(model: str, options: dict, args: dict) -> str:
    h = hashlib.sha1()
    h.update(json.dumps([model, options, __version__], sort_keys=True, default=str).encode())
    for name in ("wavelength", "ed", "ls", "lu"):
        h.update(np.ascontiguousarray(args[name], dtype=float).tobytes())
    h.update(json.dumps([args[k] for k in ("sza", "vza", "raa", "rh", "pressure", "water_temp")]).encode())
    return h.hexdigest()


def _save_cache(path: Path, keys: list, results: list) -> None:
    names = sorted({k for r in results for k in r.params})
    params = np.array([[r.params.get(k, np.nan) for k in names] for r in results], dtype=float).reshape(len(results), len(names))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, keys=np.array(keys), wavelength=results[0].wavelength,
        rrs=np.array([r.rrs for r in results]), rrs_model=np.array([r.rrs_model for r in results]),
        rrs_surface=np.array([r.rrs_surface for r in results]), cost=np.array([r.cost for r in results]),
        success=np.array([r.success for r in results]), nfev=np.array([r.nfev for r in results]),
        message=np.array([r.message for r in results]), param_names=np.array(names, dtype=str), params=params)
    tmp.replace(path)


def _load_cache(path: Path) -> dict:
    try:
        with np.load(path, allow_pickle=False) as z:
            names = list(z["param_names"])
            out = {}
            for i, key in enumerate(z["keys"]):
                out[str(key)] = ThreeCResult(
                    wavelength=z["wavelength"], rrs=z["rrs"][i], rrs_model=z["rrs_model"][i],
                    rrs_surface=z["rrs_surface"][i], params={k: float(v) for k, v in zip(names, z["params"][i])},
                    cost=float(z["cost"][i]), success=bool(z["success"][i]), message=str(z["message"][i]),
                    nfev=int(z["nfev"][i]))
            return out
    except (OSError, KeyError, ValueError):
        return {}


def run_fits(tasks: list, settings: Settings, cache_dir: Path | None = None) -> dict:
    """Fit all tasks; returns {key: ThreeCResult}.

    ``tasks`` are dicts with ``key``, ``day`` (cache group, for example ``GFN_20220618``) and
    ``args`` (keyword arguments of the fit). Groups whose fits are all in the cache are skipped.
    """
    results = {}
    groups: dict[str, list] = {}
    for t in tasks:
        groups.setdefault(t["day"], []).append(t)
    if cache_dir is not None and settings.resume:
        for day in groups:
            results.update(_load_cache(cache_dir / f"{day}.npz"))
    pending = [t for t in tasks if t["key"] not in results]
    remaining = {}
    for t in pending:
        remaining[t["day"]] = remaining.get(t["day"], 0) + 1

    def finished(day):
        if cache_dir is not None:
            keys = [t["key"] for t in groups[day]]
            _save_cache(cache_dir / f"{day}.npz", keys, [results[k] for k in keys])

    if not pending:
        return results
    bar = None
    if settings.progress:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(pending), desc=f"3C fits ({settings.threec})", unit="fit", dynamic_ncols=True)
    options = dict(settings.threec_options)
    try:
        if int(settings.workers) <= 1:
            init_worker(settings.threec, options)
            for t in pending:
                results[t["key"]] = fit_one(t["args"])
                remaining[t["day"]] -= 1
                if remaining[t["day"]] == 0:
                    finished(t["day"])
                if bar is not None:
                    bar.update(1)
        else:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=int(settings.workers), mp_context=ctx, initializer=init_worker,
                                     initargs=(settings.threec, options)) as ex:
                queue = iter(pending)
                running = {}
                while True:
                    while len(running) < 4 * int(settings.workers):
                        t = next(queue, None)
                        if t is None:
                            break
                        running[ex.submit(fit_one, t["args"])] = t
                    if not running:
                        break
                    done, _ = wait(running, return_when=FIRST_COMPLETED)
                    for fut in done:
                        t = running.pop(fut)
                        results[t["key"]] = fut.result()
                        remaining[t["day"]] -= 1
                        if remaining[t["day"]] == 0:
                            finished(t["day"])
                        if bar is not None:
                            bar.update(1)
    finally:
        if bar is not None:
            bar.close()
    return results


# per station --------------------------------------------------------------------------------------

def _base_table(sp: Spectra, settings: Settings, meteo_table) -> pd.DataFrame:
    st = sp.attrs["station_info"]
    if not (np.isfinite(st.lat) and np.isfinite(st.lon)):
        raise ValueError(f"Station {sp.station} has no coordinates: pass station=custom_station(...)")
    meta = sp.meta
    t = pd.DatetimeIndex(meta["time_utc"])
    local = t.tz_convert(st.timezone)
    geo = solar_angles(t, st.lat, st.lon)
    met = meteo_for(sp, meteo_table, rh=settings.rh, water_temp=settings.water_temp, pressure=settings.pressure,
                    tolerance_min=settings.meteo_tolerance_min)
    flags = l1_flags(sp, sza_max=settings.sza_max, sza=geo["sza"].to_numpy())
    base = pd.DataFrame({
        "station": sp.station, "file": meta["file"].to_numpy(), "time_utc": t,
        "time_local": [x.isoformat() for x in local], "date_local": [x.date().isoformat() for x in local],
        "sza": geo["sza"].to_numpy(), "saa": geo["saa"].to_numpy(),
        "vza": float(st.view_zenith_deg), "raa": float(st.relative_azimuth_deg),
        "rh_percent": met["rh_percent"].to_numpy(), "pressure_hpa": met["pressure_hpa"].to_numpy(),
        "water_temperature_c": met["water_temperature_c"].to_numpy(), "meteo_source": met["meteo_source"].to_numpy(),
    })
    for col in ("ed_qual_ok", "ls_qual_ok", "lu_qual_ok"):
        base[col] = meta[col].to_numpy(dtype=bool)
    for col in ("qual_ok", "spectra_finite", "spectra_positive", "sza_ok", "l1_ok"):
        base[col] = flags[col].to_numpy(dtype=bool)
    return base


def _tasks(sp: Spectra, base: pd.DataFrame, fit_mask: np.ndarray, settings: Settings) -> list:
    tasks = []
    for i in np.flatnonzero(fit_mask):
        r = base.iloc[i]
        args = {"wavelength": sp.wavelength, "ed": sp.ed[i], "ls": sp.ls[i], "lu": sp.lu[i],
                "sza": float(r["sza"]), "vza": float(r["vza"]), "raa": float(r["raa"]),
                "rh": float(r["rh_percent"]), "pressure": float(r["pressure_hpa"]),
                "water_temp": float(r["water_temperature_c"])}
        key = _task_key(settings.threec, settings.threec_options, args)
        tasks.append({"key": key, "day": f"{sp.station}_{r['date_local'].replace('-', '')}", "args": args, "index": int(i)})
    return tasks


def _threec_columns(base: pd.DataFrame, tasks: list, results: dict, settings: Settings, not_fitted: np.ndarray):
    n = len(base)
    rrs = np.full((n, MODEL_GRID.size), np.nan)
    cols = {"threec_model": settings.threec, "threec_fitted": np.zeros(n, dtype=bool),
            "threec_converged": np.zeros(n, dtype=bool), "threec_cost": np.full(n, np.nan),
            "threec_nfev": np.zeros(n, dtype=int), "threec_message": np.asarray(not_fitted, dtype=object)}
    params = {}
    for t in tasks:
        i, res = t["index"], results[t["key"]]
        rrs[i] = res.rrs
        cols["threec_fitted"][i] = True
        cols["threec_converged"][i] = bool(res.success)
        cols["threec_cost"][i] = res.cost
        cols["threec_nfev"][i] = res.nfev
        cols["threec_message"][i] = res.message
        for k, v in res.params.items():
            params.setdefault(f"p_{k}", np.full(n, np.nan))[i] = v
    out = pd.DataFrame(cols, index=base.index)
    out.insert(2, "threec_ok", np.isfinite(rrs).all(axis=1))
    for k, v in params.items():
        out[k] = v
    return out, rrs


def _illumination(sp: Spectra, base: pd.DataFrame, illum: IlluminationSettings, edmax_table):
    st = sp.attrs["station_info"]
    ed_df, actual_wl = ed_series(sp, illum, st)
    edmax = edmax_by_date(ed_df, st, settings=illum, table=edmax_table)
    model = modelled_ed(ed_df, st, illum, edmax=edmax)
    centred = mape_centred(model, illum)
    daily = daily_metrics(model, edmax)
    daily.insert(0, "station", sp.station)
    daily["actual_wl"] = actual_wl
    key = pd.DatetimeIndex(base["time_utc"]).floor("us")
    c = centred.set_index("time_utc")
    take = lambda col: c[col].reindex(key).to_numpy()  # noqa: E731
    cols = pd.DataFrame({
        "ed560": take("ed"), "ed560_model": take("ed_model").astype(float),
        "ed560_rel_error_percent": take("rel_error_percent").astype(float), "ed_max": take("ed_max").astype(float),
        "illum_n_points": pd.Series(take("n_points_window")).fillna(0).astype(int).to_numpy(),
        "illum_mape": take("mape_window").astype(float),
        "illum_reason": pd.Series(take("reason")).fillna("NO_ED").to_numpy(dtype=object),
    }, index=base.index)
    cols["illumination_ok"] = pd.Series(take("decision"), index=base.index).eq("retain").to_numpy()
    return cols, model, daily


def _pairs_and_summary(base: pd.DataFrame, meas: pd.DataFrame, overpasses: pd.DataFrame, model: pd.DataFrame,
                       sp: Spectra, settings: Settings, illum: IlluminationSettings):
    st = sp.attrs["station_info"]
    context = max(float(settings.context_min), float(settings.window_min))
    pr = pair(base, overpasses, window_min=context)
    if pr.empty:
        return pr, pd.DataFrame()
    used = overpasses.loc[sorted(pr["overpass_id"].unique())]
    ill = mape_window(model, used["time_utc"], st, illum)
    ill.index = used.index
    pr["in_window"] = pr["abs_delta_min"] <= float(settings.window_min)
    pr = pr.drop(columns=["is_closest", "station", "time_utc"])
    for src, dst in (("ed_max", "illum_ed_max"), ("mape_window", "illum_mape_window"), ("mape_day", "illum_mape_day"),
                     ("n_points_window", "illum_n_points_window"), ("decision", "illum_decision"),
                     ("reason", "illum_reason_overpass")):
        pr[dst] = pr["overpass_id"].map(ill[src].astype(object))
    pr["illumination_ok"] = pr["illum_decision"].eq("retain").to_numpy()
    drop = ["illumination_ok", "insitu_usable", "reason", "illum_n_points", "illum_mape", "illum_reason"]
    m = meas.drop(columns=[c for c in drop if c in meas.columns])
    pr = pr.merge(m, left_on="measurement_index", right_index=True, how="left")
    add_usable(pr, {**MEASUREMENT_CHECKS, **ILLUMINATION_CHECK})

    def nearest(mask):
        flag = np.zeros(len(pr), dtype=bool)
        sub = pr[mask]
        if len(sub):
            flag[pr.index.get_indexer(sub.groupby("overpass_id")["abs_delta_min"].idxmin().to_numpy())] = True
        return flag

    pr["is_closest"] = nearest(pr["in_window"])
    pr["is_closest_usable"] = nearest(pr["in_window"] & pr["insitu_usable"])
    first = ["station", "overpass_id", "product", "platform", "sensor", "overpass_time_utc", "time_utc", "time_local",
             "delta_min", "abs_delta_min", "in_window", "is_closest", "is_closest_usable", "insitu_usable", "reason",
             "illumination_ok", "illum_decision", "illum_reason_overpass", "illum_mape_window", "illum_mape_day",
             "illum_n_points_window", "illum_ed_max"]
    pr = pr[first + [c for c in pr.columns if c not in first]].sort_values(["overpass_time_utc", "time_utc"])
    pr = pr.reset_index(drop=True)

    rows = []
    for oid, g in pr.groupby("overpass_id", sort=False):
        o = used.loc[oid]
        win = g[g["in_window"]]
        closest = win[win["is_closest"]]
        usable = win[win["is_closest_usable"]]
        row = {"station": sp.station, "overpass_id": oid, "product": o["product"], "platform": o["platform"],
               "sensor": o["sensor"], "overpass_time_utc": o["time_utc"], "n_context": len(g), "n_in_window": len(win),
               "n_usable_in_window": int(win["insitu_usable"].sum()),
               "illum_decision": ill.loc[oid, "decision"], "illum_reason": ill.loc[oid, "reason"],
               "illum_mape_window": ill.loc[oid, "mape_window"], "illum_mape_day": ill.loc[oid, "mape_day"],
               "illum_n_points_window": ill.loc[oid, "n_points_window"], "illum_ed_max": ill.loc[oid, "ed_max"],
               "closest_time_utc": closest["time_utc"].iloc[0] if len(closest) else pd.NaT,
               "closest_delta_min": closest["delta_min"].iloc[0] if len(closest) else np.nan,
               "closest_usable": bool(closest["insitu_usable"].iloc[0]) if len(closest) else False,
               "closest_reason": closest["reason"].iloc[0] if len(closest) else "NO_MEASUREMENT_IN_WINDOW",
               "matchup_usable": bool(len(usable)),
               "usable_time_utc": usable["time_utc"].iloc[0] if len(usable) else pd.NaT,
               "usable_delta_min": usable["delta_min"].iloc[0] if len(usable) else np.nan}
        if len(usable) and o["platform"]:
            row.update(platform_bands(usable.iloc[0], o["platform"]))
        rows.append(row)
    summary = pd.DataFrame(rows)
    for col in ("overpass_time_utc", "closest_time_utc", "usable_time_utc"):
        summary[col] = pd.to_datetime(summary[col], utc=True)
    return pr, summary


def process_station(sp: Spectra, settings: Settings, overpasses: pd.DataFrame, results: dict, tasks: list,
                    base: pd.DataFrame, not_fitted: np.ndarray, edmax_table=None) -> dict:
    """Tables of one station from its 3C results (see :func:`process`)."""
    illum = settings.illumination()
    three, rrs = _threec_columns(base, tasks, results, settings, not_fitted)
    q = qwip(MODEL_GRID.astype(float), rrs, threshold=settings.qwip_max)
    ill, model, daily = _illumination(sp, base, illum, edmax_table)
    meas = pd.concat([base, three], axis=1)
    meas = pd.concat([meas, q.set_index(meas.index), ill], axis=1)
    add_usable(meas, {**MEASUREMENT_CHECKS, **ILLUMINATION_CHECK})
    sensors = list(settings.sensors) if settings.sensors else available_sensors()
    bands = resample(MODEL_GRID.astype(float), rrs, sensors, min_coverage=settings.min_coverage)
    bands = bands[band_columns(bands.columns)].set_index(meas.index)
    meas = pd.concat([meas, bands], axis=1)
    hyper = pd.concat([meas[["station", "time_utc"]],
                       pd.DataFrame(rrs, columns=[f"rrs_{w}" for w in MODEL_GRID], index=meas.index)], axis=1)
    pairs, summary = (pd.DataFrame(), pd.DataFrame())
    if overpasses is not None and len(overpasses):
        pairs, summary = _pairs_and_summary(base, meas, overpasses, model, sp, settings, illum)
    return {"measurements": meas, "pairs": pairs, "summary": summary, "hyperspectral": hyper, "daily": daily}


def process(l1, out_dir, settings: Settings | None = None, products=None, times=None, meteo=None, edmax=None,
            station=None) -> dict:
    """Process the L1 files ``l1`` (folder, file or list) and write the outputs to ``out_dir``.

    ``products`` are Sentinel-2 or Sentinel-3 product names (a text file, a folder or a list) and
    ``times`` a CSV file of overpass times. Without either, each measurement is checked in a window
    centred on itself and no pairs or summary are written. ``meteo`` is a CSV file of RH, pressure
    and water temperature, ``edmax`` a CSV file of EdMax per date and ``station`` a station for all
    files. Returns the tables as DataFrames.
    """
    s = settings or Settings()
    if s.fit not in ("all", "window"):
        raise ValueError("fit must be 'all' or 'window'")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spectra = read_l1_folder(l1, station=station)
    if not spectra:
        raise ValueError(f"No L1 measurements found in {l1}")
    overpasses = load_overpasses(products, times)
    meteo_table = read_meteo_csv(meteo) if meteo else None
    edmax_table = read_edmax_csv(edmax) if edmax else None

    prepared = []
    all_tasks = []
    for sp in spectra:
        base = _base_table(sp, s, meteo_table)
        fit_mask = base["spectra_finite"].to_numpy() & base["spectra_positive"].to_numpy() & (base["sza"].to_numpy() < 90)
        not_fitted = np.where(~(base["spectra_finite"] & base["spectra_positive"]).to_numpy(), "not fitted: L1 spectra",
                              np.where(base["sza"].to_numpy() >= 90, "not fitted: sun below the horizon", ""))
        if s.fit == "window":
            near = np.zeros(len(base), dtype=bool)
            if len(overpasses):
                near[pair(base, overpasses, window_min=s.window_min)["measurement_index"].unique()] = True
            not_fitted = np.where(fit_mask & ~near, "not fitted: outside the matchup windows", not_fitted)
            fit_mask &= near
        tasks = _tasks(sp, base, fit_mask, s)
        prepared.append((sp, base, tasks, not_fitted.astype(object)))
        all_tasks += tasks

    results = run_fits(all_tasks, s, cache_dir=out / "cache" / s.threec)
    parts = [process_station(sp, s, overpasses, results, tasks, base, nf, edmax_table)
             for sp, base, tasks, nf in prepared]
    tables = {name: pd.concat([p[name] for p in parts if len(p[name])], ignore_index=True)
              if any(len(p[name]) for p in parts) else pd.DataFrame() for name in parts[0]}

    for name, file in TABLES.items():
        if len(tables[name]):
            write_csv(tables[name], out / file)
    run = {
        "package": "wispstation-rrs", "version": __version__, "settings": asdict(s),
        "illumination": asdict(s.illumination()),
        "sensors": list(s.sensors) if s.sensors else available_sensors(),
        "inputs": {"l1": str(l1), "products": None if products is None else [str(p) for p in np.atleast_1d(products)],
                   "times": None if times is None else str(times), "meteo": None if meteo is None else str(meteo),
                   "edmax": None if edmax is None else str(edmax), "station": None if station is None else str(station)},
        "files": {sp.station: sp.attrs.get("files", []) for sp in spectra},
        "counts": {"measurements": int(len(tables["measurements"])),
                   "fitted": int(tables["measurements"]["threec_fitted"].sum()),
                   "usable": int(tables["measurements"]["insitu_usable"].sum()),
                   "overpasses": int(len(tables["summary"])),
                   "matchups_usable": int(tables["summary"]["matchup_usable"].sum()) if len(tables["summary"]) else 0},
    }
    (out / "run_settings.json").write_text(json.dumps(run, indent=1, default=str), encoding="utf-8")
    if s.plots:
        from .plots import plot_run
        plot_run(tables, out / "plots", s)
    return tables
