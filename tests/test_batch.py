"""Tests of the batch processing, the output tables, the plots and the command line."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import wispstation_rrs.batch as batch
from wispstation_rrs.threec.base import failed_result
from wispstation_rrs.batch import Settings, load_overpasses, process
from wispstation_rrs.cli import main
from wispstation_rrs.plots import plot_folder
from wispstation_rrs.table import read_table

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "examples"
TRS = "WISPstation012_20250710_L1_SPEC.nc"
WAVELENGTHS = range(350, 901)


@pytest.fixture(scope="module")
def l1(tmp_path_factory):
    folder = tmp_path_factory.mktemp("l1")
    shutil.copy(EXAMPLES / "l1" / TRS, folder / TRS)
    return folder


@pytest.fixture(scope="module")
def run(l1, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    s = Settings(fit="window", workers=1, progress=False)
    tables = process(l1, out, s, products=EXAMPLES / "overpasses.txt", meteo=EXAMPLES / "meteo.csv",
                     edmax=EXAMPLES / "edmax.csv")
    return out, tables


def test_outputs_written(run):
    out, tables = run
    for name in ("insitu_measurements.csv", "insitu_overpass_pairs.csv", "overpass_summary.csv",
                 "rrs_hyperspectral.csv", "illumination_daily.csv", "run_settings.json"):
        assert (out / name).is_file(), name
    settings = json.loads((out / "run_settings.json").read_text(encoding="utf-8"))
    assert settings["settings"]["threec"] == "jax" and settings["counts"]["fitted"] == 2
    assert settings["sensors"] == ["S2A", "S2B", "S2C", "S3A", "S3B"]
    plots = sorted(p.name for p in (out / "plots").iterdir())
    assert plots == ["ed560_TRS012.png", "illumination_TRS012_20250710T1000_S2A.png", "qwip_TRS012.png",
                     "spectra_TRS012_20250710T1000_S2A.png"]


def test_summary_and_pairs(run):
    out, tables = run
    s = read_table(out / "overpass_summary.csv")
    assert len(s) == 1 and s["tile"].item() == "T32TQN"
    o = s.iloc[0]
    assert o["platform"] == "S2A" and o["illum_decision"] == "retain"
    assert o["illum_mape_window"] == pytest.approx(0.49907432396665025, rel=1e-9)
    assert o["n_in_window"] == 2 and o["n_usable_in_window"] == 2 and o["matchup_usable"]
    assert o["usable_time_utc"] == pd.Timestamp("2025-07-10 10:00:05", tz="UTC")
    assert np.isfinite(o["rrs_B2"]) and np.isnan(o["rrs_B12"])
    p = read_table(out / "insitu_overpass_pairs.csv")
    assert len(p) == 8 and p["in_window"].sum() == 2
    assert p["is_closest"].sum() == 1 and p["is_closest_usable"].sum() == 1
    # measurements outside the window are listed but were not fitted
    outside = p[~p["in_window"]]
    assert not outside["insitu_usable"].any() and outside["reason"].str.contains("THREEC").all()
    assert (outside["threec_message"] == "not fitted: outside the matchup windows").all()
    assert p.loc[p["is_closest"], "rrs_S2A_B3"].item() == pytest.approx(o["rrs_B3"])


def test_rrs_as_paper_processing(run):
    """The JAX run with pvlib geometry and the example meteo values gives the archived Rrs."""
    out, tables = run
    h = read_table(out / "rrs_hyperspectral.csv").set_index("time_utc")
    ref = pd.read_csv(HERE / "data" / "threec_reference_trasimeno012_20250710.csv")
    for _, r in ref.iterrows():
        t = pd.Timestamp(r["time_utc"])
        if t not in h.index or not np.isfinite(h.loc[t, "rrs_500"]):
            continue
        a = h.loc[t, [f"rrs_{w}" for w in WAVELENGTHS]].to_numpy(float)
        b = r[[f"Rrs_{w}" for w in WAVELENGTHS]].to_numpy(float)
        assert np.max(np.abs(a - b)) < 1e-7


def test_measurement_table(run):
    out, tables = run
    m = read_table(out / "insitu_measurements.csv")
    assert len(m) == 45 and m["threec_fitted"].sum() == 2
    assert set(m["meteo_source"]) == {"csv"}
    assert m["insitu_usable"].sum() == 2
    assert {"qual_ok", "spectra_finite", "spectra_positive", "sza_ok", "threec_ok", "qwip_ok", "illumination_ok",
            "insitu_usable", "reason", "rrs_S3B_Oa08", "rrs_S2C_B8A"} <= set(m.columns)
    assert m["vza"].eq(45).all() and m["raa"].eq(135).all()


def test_resume_takes_the_cache(run, l1, monkeypatch):
    out, tables = run

    def fail(args):
        raise AssertionError("fit called although the result is in the cache")

    monkeypatch.setattr(batch, "fit_one", fail)
    again = process(l1, out, Settings(fit="window", workers=1, progress=False, plots=False),
                    products=EXAMPLES / "overpasses.txt", meteo=EXAMPLES / "meteo.csv", edmax=EXAMPLES / "edmax.csv")
    pd.testing.assert_frame_equal(again["hyperspectral"], tables["hyperspectral"])


def test_plot_folder(run):
    out, _ = run
    shutil.rmtree(out / "plots")
    files = plot_folder(out)
    assert len(files) == 4 and all(f.is_file() for f in files)


def test_without_overpasses(l1, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "fit_one", lambda args: calls.append(1) or failed_result("skipped"))
    tables = process(l1, tmp_path, Settings(workers=1, progress=False, plots=False))
    assert not (tmp_path / "insitu_overpass_pairs.csv").exists()
    m = tables["measurements"]
    assert len(calls) == m["threec_fitted"].sum() > 0
    assert m["illum_n_points"].max() >= 1 and not m["threec_ok"].any()


def test_pool_gives_the_in_process_result(l1, tmp_path):
    s = Settings(fit="window", workers=2, progress=False, plots=False, resume=False)
    tables = process(l1, tmp_path, s, products=EXAMPLES / "overpasses.txt", meteo=EXAMPLES / "meteo.csv")
    one = process(l1, tmp_path / "one", Settings(fit="window", workers=1, progress=False, plots=False),
                  products=EXAMPLES / "overpasses.txt", meteo=EXAMPLES / "meteo.csv")
    pd.testing.assert_frame_equal(tables["hyperspectral"], one["hyperspectral"])


def test_cli(l1, tmp_path, capsys):
    assert main(["overpasses", str(EXAMPLES / "overpasses.txt")]) == 0
    text = capsys.readouterr().out
    assert "S2B" in text and "2022-06-18T10:25:59+00:00" in text
    assert main(["overpasses", str(EXAMPLES / "overpasses.txt"), "-o", str(tmp_path / "ov.csv")]) == 0
    assert len(pd.read_csv(tmp_path / "ov.csv")) == 3
    rc = main(["process", str(l1), "-o", str(tmp_path / "out"), "--products", str(EXAMPLES / "overpasses.txt"),
               "--fit", "window", "--workers", "1", "--no-plots", "--quiet", "--sensors", "S2A"])
    assert rc == 0
    m = pd.read_csv(tmp_path / "out" / "insitu_measurements.csv")
    assert any(c.startswith("rrs_S2A_") for c in m.columns) and not any(c.startswith("rrs_S2B_") for c in m.columns)
    # no meteo file: defaults
    assert set(m["meteo_source"]) == {"default"} and m["rh_percent"].eq(60).all()


def test_load_overpasses_names_and_times(tmp_path):
    path = tmp_path / "times.csv"
    path.write_text("time_utc,platform\n2025-07-10T09:31:31Z,S3A\n")
    ov = load_overpasses(["S2A_MSIL1C_20250710T100051_N0511_R122_T32TQN_20250710T104327"], path)
    assert list(ov["platform"]) == ["S3A", "S2A"]


def test_cache_keeps_earlier_fits(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "fit_one", lambda args: calls.append(args["sza"]) or failed_result("x"))
    s = Settings(workers=1, progress=False)
    def tasks(keys):
        return [{"key": k, "day": "GFN_20220618", "args": {"sza": k}} for k in keys]
    batch.run_fits(tasks(["a", "b", "c"]), s, cache_dir=tmp_path)
    batch.run_fits(tasks(["a", "d"]), s, cache_dir=tmp_path)
    batch.run_fits(tasks(["a", "b", "c", "d"]), s, cache_dir=tmp_path)
    assert calls == ["a", "b", "c", "d"]


def test_illumination_window_apart_from_matchup_window(l1, tmp_path):
    common = dict(products=EXAMPLES / "overpasses.txt", meteo=EXAMPLES / "meteo.csv", edmax=EXAMPLES / "edmax.csv")
    wide = process(l1, tmp_path / "a", Settings(fit="window", window_min=30, workers=1, progress=False, plots=False),
                   **common)
    o = wide["summary"].iloc[0]
    assert o["n_in_window"] == 4
    assert o["illum_n_points_window"] == 2
    assert o["illum_mape_window"] == pytest.approx(0.49907432396665025, rel=1e-9)


def test_cli_new_station(l1, tmp_path):
    folder = tmp_path / "l1"
    folder.mkdir()
    shutil.copy(l1 / TRS, folder / "WISPstation077_20250710_L1_SPEC.nc")
    rc = main(["process", str(folder), "-o", str(tmp_path / "out"), "--station", "TRS077", "--lat", "43.123064",
               "--lon", "12.129649", "--altitude", "258", "--timezone", "Europe/Rome", "--tiles", "T32TQN",
               "--products", str(EXAMPLES / "overpasses.txt"), "--fit", "window", "--workers", "1", "--no-plots",
               "--quiet"])
    assert rc == 0
    s = pd.read_csv(tmp_path / "out" / "overpass_summary.csv")
    assert list(s["station"]) == ["TRS077"] and s["matchup_usable"].item()
    with pytest.raises(SystemExit):
        main(["process", str(folder), "--lat", "43.1", "--lon", "12.1"])
