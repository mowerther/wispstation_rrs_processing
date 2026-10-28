"""Command line: ``wisp-rrs process``, ``wisp-rrs overpasses`` and ``wisp-rrs plot``."""
from __future__ import annotations

import argparse
import sys

from . import __version__


def _process_parser(sub):
    p = sub.add_parser("process", help="compute Rrs and quality criteria from a folder of L1 files")
    p.add_argument("l1", nargs="+", help="L1 folder(s) or file(s)")
    p.add_argument("-o", "--out", default="wisp_rrs_output", help="output folder (default: wisp_rrs_output)")
    p.add_argument("--threec", choices=["jax", "o25"], default="jax",
                   help="3C implementation: jax (Werther, paper processing) or o25 (Pitarch 2026), default jax")
    p.add_argument("--station", help="station code for all files (default: from the file names)")
    g = p.add_argument_group("new station", "with --station CODE, for a station that is not in the station table")
    g.add_argument("--lat", type=float, help="latitude in degrees north")
    g.add_argument("--lon", type=float, help="longitude in degrees east")
    g.add_argument("--altitude", type=float, default=0.0, help="altitude of the lake surface in m (default 0)")
    g.add_argument("--timezone", default="UTC", help="local time zone, for example Europe/Zurich (default UTC)")
    g.add_argument("--tiles", nargs="+", default=(), help="Sentinel-2 tiles that cover the station, for example T32TMT")
    g = p.add_argument_group("matchup window")
    g.add_argument("--products", nargs="+", help="Sentinel-2 or Sentinel-3 product names: a text file, a folder or names")
    g.add_argument("--times", help="CSV file of overpass times (time_utc, optional platform and label)")
    g.add_argument("--window", type=float, default=15.0, help="matchup window in minutes (default 15)")
    g.add_argument("--illum-window", type=float, default=15.0,
                   help="window of the illumination MAPE in minutes (default 15)")
    g.add_argument("--context", type=float, default=60.0,
                   help="measurements within this many minutes of an overpass are listed in the pairs table (default 60)")
    g.add_argument("--fit", choices=["all", "window"], default="all",
                   help="fit all measurements or only those in a matchup window (default all)")
    g = p.add_argument_group("meteorological inputs")
    g.add_argument("--meteo", help="CSV file with time_utc, station, rh_percent, pressure_hpa, water_temperature_c")
    g.add_argument("--rh", type=float, default=60.0, help="relative humidity without a meteo row, %% (default 60)")
    g.add_argument("--water-temp", type=float, default=20.0, help="water temperature without a meteo row, degC (default 20)")
    g.add_argument("--pressure", type=float, help="air pressure without a meteo row, hPa (default: from the station altitude)")
    g = p.add_argument_group("quality criteria")
    g.add_argument("--mape-max", type=float, default=20.0, help="limit of the Ed(560) MAPE, %% (default 20)")
    g.add_argument("--edmax", help="CSV file of EdMax per date (station, date_local, ed_max)")
    g.add_argument("--qwip-max", type=float, default=0.2, help="limit of the absolute QWIP score (default 0.2)")
    g.add_argument("--sza-max", type=float, help="limit of the solar zenith angle in degrees (default: none)")
    g = p.add_argument_group("bands")
    g.add_argument("--sensors", nargs="+", help="sensors of the band Rrs (default: S2A S2B S2C S3A S3B)")
    g.add_argument("--min-coverage", type=float, default=0.99,
                   help="minimum share of a band response covered by the spectrum (default 0.99)")
    g = p.add_argument_group("run")
    g.add_argument("--workers", type=int, default=2, help="worker processes of the 3C fits (default 2)")
    g.add_argument("--no-resume", action="store_true", help="fit again instead of using the results of earlier runs")
    g.add_argument("--no-plots", action="store_true", help="write no plots")
    g.add_argument("--quiet", action="store_true", help="no progress bar")
    return p


def _run_process(a) -> int:
    from .batch import Settings, process

    station = a.station
    if a.lat is not None or a.lon is not None:
        from .stations import custom_station

        if a.station is None or a.lat is None or a.lon is None:
            raise SystemExit("a new station needs --station, --lat and --lon")
        station = custom_station(a.station, a.lat, a.lon, a.altitude, timezone=a.timezone, tiles=a.tiles)
    s = Settings(threec=a.threec, window_min=a.window, illum_window_min=a.illum_window, context_min=a.context,
                 rh=a.rh, water_temp=a.water_temp,
                 pressure=a.pressure, sensors=tuple(a.sensors) if a.sensors else None, min_coverage=a.min_coverage,
                 qwip_max=a.qwip_max, mape_max=a.mape_max, sza_max=a.sza_max, fit=a.fit, workers=a.workers,
                 resume=not a.no_resume, plots=not a.no_plots, progress=not a.quiet)
    l1 = a.l1 if len(a.l1) > 1 else a.l1[0]
    tables = process(l1, a.out, s, products=a.products, times=a.times, meteo=a.meteo, edmax=a.edmax,
                     station=station)
    m = tables["measurements"]
    print(f"{len(m)} measurements, {int(m['threec_fitted'].sum())} fitted, {int(m['insitu_usable'].sum())} usable")
    if len(tables["summary"]):
        o = tables["summary"]
        print(f"{len(o)} overpasses with measurements, {int(o['matchup_usable'].sum())} with a usable measurement")
    print(f"Output: {a.out}")
    return 0


def _run_overpasses(a) -> int:
    from .batch import load_overpasses
    from .table import write_csv

    ov = load_overpasses(a.products, a.times)
    if a.out:
        write_csv(ov, a.out)
    else:
        ov = ov.assign(time_utc=ov["time_utc"].map(lambda t: t.isoformat()))
        print(ov.to_string(index=False))
    return 0


def _run_plot(a) -> int:
    from .plots import plot_folder

    files = plot_folder(a.out)
    print(f"{len(files)} plots in {a.out}/plots")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="wisp-rrs", description="Rrs and quality criteria from WISPstation L1 files.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    _process_parser(sub)
    p = sub.add_parser("overpasses", help="list the overpass times of product names or a times CSV")
    p.add_argument("products", nargs="*", help="product names, a text file or a folder")
    p.add_argument("--times", help="CSV file of overpass times")
    p.add_argument("-o", "--out", help="write the table to this CSV file")
    p = sub.add_parser("plot", help="make the plots of an output folder again")
    p.add_argument("out", help="output folder of wisp-rrs process")
    a = parser.parse_args(argv)
    if a.command == "process":
        return _run_process(a)
    if a.command == "overpasses":
        if not a.products and not a.times:
            parser.error("give product names or --times")
        return _run_overpasses(a)
    return _run_plot(a)


if __name__ == "__main__":
    sys.exit(main())
