"""Tests of the sun position and of the solar schedule of the illumination check."""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wispstation_rrs.geometry import solar_angles, solar_schedule_table
from wispstation_rrs.stations import STATIONS

DATA = Path(__file__).resolve().parent / "data"
CASES = {"greifensee_20220618": "GFN", "geneva_20250528": "LXP", "trasimeno012_20250710": "TRS012"}


@pytest.mark.parametrize("key,code", CASES.items())
def test_zenith_close_to_paper_processing(key, code):
    ref = pd.read_csv(DATA / f"threec_reference_{key}.csv", usecols=["time_utc", "sza"])
    st = STATIONS[code]
    a = solar_angles(ref["time_utc"], st.lat, st.lon)
    # pvlib apparent zenith against the solar zenith of the paper processing
    assert np.max(np.abs(a["sza"].to_numpy() - ref["sza"].to_numpy())) < 0.01
    assert np.all(a["sza_true"] > a["sza"])


def test_solar_angles_table():
    st = STATIONS["GFN"]
    a = solar_angles(pd.DatetimeIndex(["2022-06-18 11:25", "2022-06-18 23:00"], tz="UTC"), st.lat, st.lon)
    assert list(a.columns) == ["time_utc", "sza", "sza_true", "saa"]
    # near solar noon the sun stands about south, at midnight below the horizon
    assert 170 < a["saa"].iloc[0] < 190 and a["sza"].iloc[0] < 30
    assert a["sza"].iloc[1] > 90


def test_schedule_as_paper_tables():
    gfn = solar_schedule_table([date(2022, 6, 18)], STATIONS["GFN"].lat, STATIONS["GFN"].lon, "Europe/Zurich").iloc[0]
    assert gfn["sunrise_local"] == pd.Timestamp("2022-06-18 05:28:12")
    assert gfn["sunset_local"] == pd.Timestamp("2022-06-18 21:24:55")
    assert gfn["day_length_s"] == gfn["sunset_sec"] - gfn["sunrise_sec"]
    trs = solar_schedule_table([date(2025, 7, 10)], STATIONS["TRS012"].lat, STATIONS["TRS012"].lon, "Europe/Rome").iloc[0]
    assert trs["sunrise_local"] == pd.Timestamp("2025-07-10 05:42:05")
    assert trs["sunset_local"] == pd.Timestamp("2025-07-10 20:51:50")
    noon = trs["noon_utc"]
    assert pd.Timestamp("2025-07-10 11:10", tz="UTC") < noon < pd.Timestamp("2025-07-10 11:20", tz="UTC")
