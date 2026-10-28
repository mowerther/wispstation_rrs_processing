"""Tests of the L1 readers, the station table and the meteo input."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wispstation_rrs.io import list_l1_files, meteo_for, read_l1_file, read_l1_folder, read_meteo_csv
from wispstation_rrs.stations import (STATIONS, barometric_pressure_hpa, custom_station, detect_station,
                                      get_station, parse_l1_filename)

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "examples"
L1 = EXAMPLES / "l1"
GFN_FILE = "L1B_Lake_Greifen_Sprectral_20220618_040005.nc"


@pytest.fixture(scope="module")
def spectra():
    return {sp.station: sp for sp in read_l1_folder(L1)}


def test_file_names():
    assert parse_l1_filename(GFN_FILE) == ("GFN", "datalakes", "20220618")
    assert parse_l1_filename("L1B_Lake_Greifen_Spectral_20220618_040005.nc") == ("GFN", "datalakes", "20220618")
    assert parse_l1_filename("L1_Lexplore_Sprectral_20250528_060005.nc") == ("LXP", "datalakes", "20250528")
    assert parse_l1_filename("WISPstation012_20250710_L1_SPEC.nc") == ("TRS012", "wispcloud", "20250710")
    assert parse_l1_filename("WISPstation077_20250710_L1_SPEC.nc") == (None, "wispcloud", "20250710")
    assert parse_l1_filename("other.nc") == (None, None, None)
    assert detect_station(GFN_FILE).code == "GFN"
    assert get_station("lexplore") is STATIONS["LXP"]
    with pytest.raises(KeyError):
        get_station("nowhere")


def test_examples(spectra):
    assert sorted(spectra) == ["GFN", "LXP", "TRS012"]
    sizes = {"GFN": 45, "LXP": 38, "TRS012": 45}
    for code, sp in spectra.items():
        assert len(sp) == sizes[code]
        assert sp.wavelength.size == 551 and sp.wavelength[0] == 350 and sp.wavelength[-1] == 900
        assert sp.ed.shape == sp.ls.shape == sp.lu.shape == (sizes[code], 551)
        t = sp.meta["time_utc"]
        assert str(t.dt.tz) == "UTC" and t.is_monotonic_increasing and not t.duplicated().any()
        assert sp.meta[["ed_qual_ok", "ls_qual_ok", "lu_qual_ok"]].all().all()
        assert (sp.meta["station"] == code).all()
        assert sp.attrs["station_info"].code == code
    assert spectra["TRS012"].attrs["file_format"] == "wispcloud"
    assert spectra["GFN"].attrs["file_format"] == "datalakes"
    # measurement times of the example days (first one of each file)
    assert spectra["GFN"].meta["time_utc"].iloc[0] == pd.Timestamp("2022-06-18 04:00:05.384753942", tz="UTC")
    assert spectra["TRS012"].meta["time_utc"].iloc[0] == pd.Timestamp("2025-07-10 04:00:05", tz="UTC")


def test_values_as_in_the_file(spectra):
    import netCDF4
    with netCDF4.Dataset(L1 / GFN_FILE) as ds:
        ed = np.asarray(ds["downirr"][:], dtype=float)
        lu = np.asarray(ds["uprad"][:], dtype=float)
    sp = spectra["GFN"]
    assert np.array_equal(sp.ed, ed.T) and np.array_equal(sp.lu, lu.T)


def test_days_split_over_files_are_merged(tmp_path):
    shutil.copy(L1 / GFN_FILE, tmp_path / GFN_FILE)
    # the same day again under the other spelling, as two parts of one day would come
    shutil.copy(L1 / GFN_FILE, tmp_path / "L1B_Lake_Greifen_Spectral_20220618_120005.nc")
    (sp,) = read_l1_folder(tmp_path)
    assert len(sp) == 45
    assert sp.attrs["n_duplicates_dropped"] == 45
    assert len(sp.attrs["files"]) == 2
    # the first file in name order is kept ("Spectral" sorts before "Sprectral")
    assert (sp.meta["file"] == "L1B_Lake_Greifen_Spectral_20220618_120005.nc").all()


def test_station_override_and_errors(tmp_path):
    shutil.copy(L1 / GFN_FILE, tmp_path / "my_station.nc")
    with pytest.raises(ValueError, match="station"):
        read_l1_folder(tmp_path)
    site = custom_station("ABC", 47.0, 8.0, 400.0, timezone="Europe/Zurich")
    sp = read_l1_file(tmp_path / "my_station.nc", station=site)
    assert sp.station == "ABC" and len(sp) == 45 and sp.attrs["file_format"] == "datalakes"
    with pytest.raises(FileNotFoundError):
        list_l1_files(tmp_path / "missing")


def test_meteo_csv(spectra):
    table = read_meteo_csv(EXAMPLES / "meteo.csv")
    assert str(table["time_utc"].dt.tz) == "UTC"
    for code, sp in spectra.items():
        m = meteo_for(sp, table)
        assert (m["meteo_source"] == "csv").all(), code
    # the inputs of the archived 3C fits
    gfn = meteo_for(spectra["GFN"], table)
    row = gfn[gfn["time_utc"] == pd.Timestamp("2022-06-18 10:00:05.255028963", tz="UTC")].iloc[0]
    assert row["rh_percent"] == pytest.approx(40.85044861) and row["pressure_hpa"] == pytest.approx(1020.41)
    assert row["water_temperature_c"] == 20.0
    trs = meteo_for(spectra["TRS012"], table)
    row = trs[trs["time_utc"] == pd.Timestamp("2025-07-10 10:00:05", tz="UTC")].iloc[0]
    assert row["water_temperature_c"] == pytest.approx(25.84) and row["rh_percent"] == pytest.approx(46.18086624145508)


def test_meteo_defaults(spectra):
    sp = spectra["GFN"]
    m = meteo_for(sp)
    assert (m["meteo_source"] == "default").all()
    assert (m["rh_percent"] == 60.0).all() and (m["water_temperature_c"] == 20.0).all()
    assert m["pressure_hpa"].iloc[0] == pytest.approx(barometric_pressure_hpa(435.0))
    assert barometric_pressure_hpa(0.0) == 1013.25
    m = meteo_for(sp, rh=70.0, water_temp=15.0, pressure=1000.0)
    assert (m["rh_percent"] == 70.0).all() and (m["water_temperature_c"] == 15.0).all()
    assert (m["pressure_hpa"] == 1000.0).all()
    # one row two hours before the first measurement: outside the 90 min tolerance
    t0 = sp.meta["time_utc"].iloc[0]
    table = pd.DataFrame({"time_utc": [t0 - pd.Timedelta(hours=2)], "station": ["GFN"], "rh_percent": [50.0]})
    m = meteo_for(sp, table)
    assert m["meteo_source"].iloc[0] == "default" and m["rh_percent"].iloc[0] == 60.0
    # rows of another station are not used
    table = pd.DataFrame({"time_utc": [t0], "station": ["LXP"], "rh_percent": [50.0]})
    assert meteo_for(sp, table)["meteo_source"].iloc[0] == "default"
    # a missing value in a matched row takes the default
    table = pd.DataFrame({"time_utc": [t0], "rh_percent": [50.0]})
    m = meteo_for(sp, table)
    assert m["meteo_source"].iloc[0] == "csv" and m["rh_percent"].iloc[0] == 50.0
    assert m["water_temperature_c"].iloc[0] == 20.0
