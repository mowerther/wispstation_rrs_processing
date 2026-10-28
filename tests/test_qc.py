"""Tests of the L1 flags, QWIP and the illumination check."""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wispstation_rrs.io import read_l1_folder
from wispstation_rrs.overpass import read_products, unique_overpasses
from wispstation_rrs.qc import check_measurements, check_overpasses, l1_flags, qwip, read_edmax_csv
from wispstation_rrs.qc.illumination import build_sin_curve, edmax_by_date, ed_series
from wispstation_rrs.stations import STATIONS
from wispstation_rrs.types import Spectra

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent / "examples"
DATA = HERE / "data"

# window MAPE and day MAPE (%) of the paper tables, EdMax pooled over plus or minus 15 days
PAPER = {"GFN": (2.311069330930203, 6.467613114640451), "LXP": (80.80407108907062, 70.57195141255733),
         "TRS012": (0.49907432396665025, 21.456526014926457)}
DECISION = {"GFN": "retain", "LXP": "reject", "TRS012": "retain"}


@pytest.fixture(scope="module")
def spectra():
    return {sp.station: sp for sp in read_l1_folder(EXAMPLES / "l1")}


@pytest.fixture(scope="module")
def overpasses():
    return unique_overpasses(read_products(EXAMPLES / "overpasses.txt"))


def test_l1_flags(spectra):
    sp = spectra["GFN"]
    f = l1_flags(sp)
    assert f["l1_ok"].all()
    bad = sp.subset(np.arange(len(sp)))
    bad.ed = bad.ed.copy()
    bad.lu = bad.lu.copy()
    bad.ed[0, 100] = np.nan
    bad.lu[1, 200] = -1e-4
    bad.lu[2, 0] = -1e-4  # 350 nm is inside the checked range
    bad.meta = bad.meta.copy()
    bad.meta.loc[3, "lu_qual_ok"] = False
    f = l1_flags(bad)
    assert not f["spectra_finite"].iloc[0] and f["spectra_positive"].iloc[0] is np.False_
    assert f["spectra_finite"].iloc[1] and not f["spectra_positive"].iloc[1] and not f["spectra_positive"].iloc[2]
    assert not f["qual_ok"].iloc[3]
    assert list(np.flatnonzero(~f["l1_ok"].to_numpy())) == [0, 1, 2, 3]
    g = l1_flags(sp, sza_max=60.0)
    assert 0 < g["sza_ok"].sum() < len(sp)


def test_qwip_against_reference_functions():
    ref = pd.read_csv(DATA / "qwip_reference.csv")
    rrs = []
    for key in ("greifensee_20220618", "geneva_20250528", "trasimeno012_20250710"):
        t = pd.read_csv(DATA / f"threec_reference_{key}.csv")
        rrs.append(t[[f"Rrs_{w}" for w in range(350, 901)]].to_numpy(float))
    q = qwip(np.arange(350, 901), np.vstack(rrs))
    np.testing.assert_allclose(q["avw_nm"], ref["avw_nm"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(q["ndi"], ref["ndi"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(q["qwip_score"], ref["qwip_score"], rtol=0, atol=1e-12)
    assert list(q["qwip_ok"]) == list(np.abs(ref["qwip_score"]) <= 0.2)


def test_qwip_edge_cases():
    wl = np.arange(405.0, 701.0, 10.0)  # bands at 485 and 495 nm: none within 2 nm of 492 nm
    q = qwip(wl, np.full(wl.size, 0.01))
    assert np.isnan(q["qwip_score"].iloc[0]) and not q["qwip_ok"].iloc[0]
    wl = np.arange(350.0, 901.0)
    r = np.full((2, wl.size), 0.005)
    r[1] = np.nan
    q = qwip(wl, r)
    assert q["avw_nm"].iloc[0] == pytest.approx(1.0 / np.mean(1.0 / np.arange(400.0, 701.0)))
    assert q["ndi"].iloc[0] == 0 and not q["qwip_ok"].iloc[1]


def test_illumination_with_paper_edmax(spectra, overpasses):
    table = read_edmax_csv(EXAMPLES / "edmax.csv")
    for code, sp in spectra.items():
        res, daily = check_overpasses(sp, overpasses, edmax_table=table)
        row = res[res["reason"] != "NO_WISP_DAY_DATA"]
        assert len(row) == 1, code
        row = row.iloc[0]
        assert row["mape_window"] == pytest.approx(PAPER[code][0], rel=1e-12)
        assert row["mape_day"] == pytest.approx(PAPER[code][1], rel=1e-12)
        assert row["n_points_window"] == 2 and row["decision"] == DECISION[code]
        assert row["actual_wl"] == 560.0
        assert daily["ed_max_source"].tolist() == ["table"]


def test_illumination_from_the_example_day_alone(spectra, overpasses):
    res, daily = check_overpasses(spectra["GFN"], overpasses)
    row = res[res["platform"] == "S2B"].iloc[0]
    assert row["decision"] == "retain" and row["mape_window"] == pytest.approx(2.3497877, rel=1e-6)
    assert daily["ed_max_source"].tolist() == ["noon"] and daily["n_noon_samples"].tolist() == [2]
    # no measurement within 15 min of solar noon on the Lake Geneva day: no EdMax
    res, daily = check_overpasses(spectra["LXP"], overpasses)
    row = res[res["platform"] == "S2C"].iloc[0]
    assert row["reason"] == "NO_MODEL" and row["decision"] == "reject"
    assert np.isnan(daily["ed_max"].iloc[0])
    # a threshold changes the decision only
    res, _ = check_overpasses(spectra["GFN"], overpasses, mape_threshold=2.0)
    assert res[res["platform"] == "S2B"]["reason"].item() == "MAPE_GT_THRESHOLD"


def _synthetic(days, noon_values):
    """Ed(560) of measurements at solar noon on ``days``, plus one at 09:00 UTC on each day."""
    st = STATIONS["GFN"]
    from wispstation_rrs.geometry import solar_schedule_table
    sched = solar_schedule_table(days, st.lat, st.lon, st.timezone)
    times, ed = [], []
    for (_, row), value in zip(sched.iterrows(), noon_values):
        times += [row["noon_utc"], pd.Timestamp(row["date_local"]).tz_localize("UTC") + pd.Timedelta(hours=9)]
        ed += [value, value / 2]
    meta = pd.DataFrame({"station": "GFN", "file": "x.nc", "time_utc": pd.DatetimeIndex(times)})
    e = np.array(ed)[:, None]
    return Spectra("GFN", np.array([560.0]), e, e, e, meta, {"station_info": st})


def test_edmax_pool():
    days = [date(2022, 6, 1) + pd.Timedelta(days=k) for k in (0, 10, 20, 40)]
    sp = _synthetic(days, [1.0, 2.0, 3.0, 4.0])
    ed_df, wl = ed_series(sp)
    assert wl == 560.0
    em = edmax_by_date(ed_df, "GFN").set_index("date_local")
    # 2022-06-11 pools 06-01, 06-11 and 06-21: 90th percentile of 1, 2, 3
    assert em.loc[days[1], "ed_max"] == pytest.approx(np.percentile([1.0, 2.0, 3.0], 90))
    assert em.loc[days[1], "n_noon_days"] == 3
    assert em.loc[days[3], "ed_max"] == 4.0 and em.loc[days[3], "n_noon_samples"] == 1
    em = edmax_by_date(ed_df, "GFN", table=pd.DataFrame({"date_local": [days[3]], "ed_max": [9.0]})).set_index("date_local")
    assert em.loc[days[3], "ed_max"] == 9.0 and em.loc[days[3], "ed_max_source"] == "table"
    assert em.loc[days[0], "ed_max_source"] == "noon"


def test_sine_model_and_centred_window():
    t, ed = build_sin_curve(3600 * 16, 2.0, 8.0, 1.8)
    assert t[0] == 480.0 and ed[0] == 0.0
    assert ed.max() == pytest.approx(2.0, rel=1e-6) and t[np.argmax(ed)] == pytest.approx(480 + 8 * 3600, abs=1)
    days = [date(2022, 6, 18)]
    sp = _synthetic(days, [1.5])
    out = check_measurements(sp)
    assert list(out.columns[-5:]) == ["n_points_window", "mape_window", "decision", "reason", "actual_wl"]
    # the noon value equals EdMax, the model at noon is close to EdMax
    noon = out.iloc[out["ed"].to_numpy().argmax()]
    assert noon["n_points_window"] == 1 and noon["mape_window"] < 1.0 and noon["decision"] == "retain"
