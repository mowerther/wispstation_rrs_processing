"""Tests of the overpass times from product names and time tables, and of the matchup window."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wispstation_rrs.overpass import pair, parse_product_name, read_products, read_times_csv, unique_overpasses

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_sentinel2_names():
    o = parse_product_name("S2B_MSIL1C_20220618T102559_N0510_R108_T32TMT_20240628T075737.SAFE")
    assert o.platform == "S2B" and o.sensor == "MSI" and o.mission == "Sentinel-2" and o.tile == "T32TMT"
    assert o.time_utc == pd.Timestamp("2022-06-18 10:25:59", tz="UTC")
    assert o.product == "S2B_MSIL1C_20220618T102559_N0510_R108_T32TMT_20240628T075737"
    o = parse_product_name("S2C_MSIL2A_20250528T102621_N0511_R108_T31TGM_20250528T123122.zip")
    assert o.platform == "S2C" and o.time_utc == pd.Timestamp("2025-05-28 10:26:21", tz="UTC")


def test_sentinel3_names():
    o = parse_product_name("S3B_OL_1_EFR____20220618T095512_20220618T095812_20220619T143005_0179_067_250_2160_PS2_O_NT_002.SEN3")
    assert o.platform == "S3B" and o.sensor == "OLCI" and o.mission == "Sentinel-3"
    assert o.time_utc == pd.Timestamp("2022-06-18 09:56:42", tz="UTC")
    o = parse_product_name("S3A_OL_2_WFR____20250710T093001_20250710T093301_20250711T171022_0179_127_307_2340_MAR_O_NT_003")
    assert o.time_utc == pd.Timestamp("2025-07-10 09:31:31", tz="UTC")


def test_bad_names():
    for name in ("S2B_MSIL1B_20220618T102559", "LC08_L1TP_194027_20220618", "S3A_SL_1_RBT____20220618T095512_20220618T095812_"):
        with pytest.raises(ValueError):
            parse_product_name(name)


def test_example_products():
    ov = read_products(EXAMPLES / "overpasses.txt")
    assert len(ov) == 4
    u = unique_overpasses(ov)
    # the two Lake Geneva tiles are one acquisition
    assert len(u) == 3
    assert list(u["platform"]) == ["S2B", "S2C", "S2A"]
    assert u["product"].iloc[1].count(";") == 1


def test_products_folder(tmp_path):
    (tmp_path / "S2A_MSIL1C_20250710T100051_N0511_R122_T32TQN_20250710T104327.SAFE").mkdir()
    (tmp_path / "notes.txt").write_text("not a product")
    ov = read_products(tmp_path)
    assert len(ov) == 1 and ov[0].platform == "S2A"


def test_times_csv(tmp_path):
    path = tmp_path / "times.csv"
    path.write_text("time_utc,platform,label\n2025-07-10T10:00:51Z,sentinel-2a,first\n"
                    "2025-07-10T09:31:31+00:00,S3B,\n")
    ov = read_times_csv(path)
    assert [o.platform for o in ov] == ["S3B", "S2A"]
    assert ov[1].sensor == "MSI" and ov[1].label == "first" and ov[0].sensor == "OLCI"
    assert ov[1].time_utc == pd.Timestamp("2025-07-10 10:00:51", tz="UTC")


def test_pair_window():
    t0 = pd.Timestamp("2025-07-10 10:00:00", tz="UTC")
    times = t0 + pd.to_timedelta([-20, -15, -4, 3, 15, 15.5], unit="min")
    meta = pd.DataFrame({"station": "TRS012", "time_utc": times})
    p = pair(meta, [t0], window_min=15)
    assert list(p["measurement_index"]) == [1, 2, 3, 4]
    assert np.allclose(p["delta_min"], [-15, -4, 3, 15])
    assert p["is_closest"].sum() == 1 and p.loc[p["is_closest"], "measurement_index"].item() == 3
    assert pair(meta, [t0], window_min=2).empty
    p = pair(meta, [t0], window_min=30)
    assert len(p) == 6
