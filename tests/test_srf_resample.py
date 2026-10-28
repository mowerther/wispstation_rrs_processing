"""Band resampling against the production MSI convolution of the paper."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import interp1d

from wispstation_rrs.srf import DATA_DIR, available_sensors, band_centres, load_srf, resample

trapezoid = getattr(np, "trapezoid", None) or np.trapz

TEST_DATA = Path(__file__).resolve().parent / "data"
MSI = ("S2A", "S2B", "S2C")
PRIMARY = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8A"]
# B8 coverage of a 350-900 nm grid (SRF area up to 900 nm over the full area)
B8_COVERAGE = {"S2A": 0.99309, "S2B": 0.99486, "S2C": 0.99078}


# Production convolution (MSI paper, trapezoid version), copied unchanged apart from layout.
def load_srf_production(srf_path):
    with open(srf_path, 'r') as f:
        for i, line in enumerate(f):
            if ('SR_WL' in line or 'Wavelength' in line) and ';' in line:
                skiprows = i; delimiter = ';'; break
            elif ('SR_WL' in line or 'Wavelength' in line):
                skiprows = i; delimiter = ','; break
        else:
            raise ValueError(f"Could not find a suitable header row in {srf_path}")
    srf_df = pd.read_csv(srf_path, skiprows=skiprows, delimiter=delimiter)
    wavelength_col = srf_df.columns[0]
    srf_df = srf_df.rename(columns={wavelength_col: 'Wavelength'})
    srf_df['Wavelength'] = pd.to_numeric(srf_df['Wavelength'], errors='coerce')
    srf_df = srf_df.dropna(subset=['Wavelength']).set_index('Wavelength'); srf_df.index = srf_df.index.astype(float)
    potential_band_cols = [c for c in srf_df.columns if 'SR_AV' in c or c.startswith('B')]
    for c in potential_band_cols: srf_df[c] = pd.to_numeric(srf_df[c], errors='coerce')
    band_cols = [c for c in potential_band_cols if pd.api.types.is_numeric_dtype(srf_df[c])]
    return srf_df[band_cols].fillna(0)


def interpolate_srf(srf_df, target_wl):
    out = {}
    for band in srf_df.columns:
        sb = srf_df[band].dropna()
        f = interp1d(sb.index.values, sb.values, kind='linear', bounds_error=False, fill_value=0)
        v = f(target_wl); v[v < 0] = 0
        out[band] = v
    return pd.DataFrame(out, index=target_wl)


def _trapz_weights(x):
    x = x.astype(float); n = x.size
    if n == 1: return np.array([1.0], dtype=float)
    dx = np.diff(x); w = np.empty_like(x)
    w[1:-1] = (dx[1:] + dx[:-1]) / 2.0; w[0] = dx[0] / 2.0; w[-1] = dx[-1] / 2.0
    return w


def convolve_rrs_reflectance(rrs_df, srf_df):
    common = rrs_df.index.intersection(srf_df.index)
    if common.size == 0:
        return pd.DataFrame(index=rrs_df.columns, columns=srf_df.columns, dtype=float)
    x = common.values.astype(float)
    R = rrs_df.loc[common].to_numpy(dtype=float)
    S = srf_df.loc[common].to_numpy(dtype=float)
    mask = np.isfinite(R).astype(float)
    R = np.nan_to_num(R, nan=0.0)
    w = _trapz_weights(x)
    Rw = (R * mask) * w[:, None]
    Mw = mask * w[:, None]
    num = Rw.T.dot(S)
    den = Mw.T.dot(S)
    out = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    return pd.DataFrame(out, index=rrs_df.columns, columns=srf_df.columns)


def production(wavelength, spectra, sensor, tmp_path):
    """Production result for spectra (n, m): Rrs on 350-2500 nm at 1 nm, ESA-format SRF file."""
    table = load_srf(sensor).pivot(index="wavelength_nm", columns="band", values="response")
    table = table[list(dict.fromkeys(load_srf(sensor)["band"]))]
    table.index = table.index.astype(int)
    table.index.name = "SR_WL"
    table.columns = [f"{sensor}_SR_AV_{b}" for b in table.columns]
    path = tmp_path / f"spectral_response_{sensor.lower()}.csv"
    if not path.exists():
        table.to_csv(path, sep=";")
    srf = interpolate_srf(load_srf_production(path), np.arange(350, 2501).astype('int64'))
    data_rrs = pd.DataFrame(np.asarray(spectra, float).T, index=np.asarray(wavelength, float))
    wl_target = np.arange(350, 2501).astype('int64')
    data_rrs_interp = data_rrs.reindex(wl_target).interpolate(method='linear')
    out = convolve_rrs_reflectance(data_rrs_interp, srf)
    out.columns = [c.replace(f"{sensor}_SR_AV_", "") for c in out.columns]
    return out.reset_index(drop=True)


def synthetic(n=25, seed=3):
    """Smooth positive spectra on the WISPstation grid (350-900 nm, 1 nm)."""
    rng = np.random.default_rng(seed)
    x = np.arange(350.0, 901.0)
    t = (x - 350.0) / 550.0
    spectra = []
    for _ in range(n):
        base = rng.uniform(0.002, 0.02) * np.exp(-((x - rng.uniform(500, 600)) / rng.uniform(60, 150)) ** 2)
        red = rng.uniform(0, 0.004) * np.exp(-((x - rng.uniform(680, 720)) / 15.0) ** 2)
        wiggle = 2e-4 * np.sin(2 * np.pi * rng.uniform(3, 12) * t + rng.uniform(0, 6))
        spectra.append(base + red + wiggle + rng.normal(0, 2e-5, x.size) + 1e-4)
    return x, np.array(spectra)


@pytest.fixture(scope="module")
def real():
    """Twelve 3C spectra (Greifensee 2025, Lake Geneva 2022-2025) and the production MSI values."""
    spectra = pd.read_csv(TEST_DATA / "srf_3c_spectra.csv")
    reference = pd.read_csv(TEST_DATA / "srf_3c_production_msi.csv")
    wl_cols = [c for c in spectra.columns if c.isdigit()]
    return np.array(wl_cols, dtype=float), spectra[wl_cols].to_numpy(float), spectra, reference


@pytest.mark.parametrize("sensor", MSI)
def test_synthetic_matches_production(sensor, tmp_path):
    x, spectra = synthetic()
    spectra[3, :12] = np.nan  # leading gap, left as NaN by both
    ours = resample(x, spectra, sensor)
    ref = production(x, spectra, sensor, tmp_path)
    for band in PRIMARY:
        np.testing.assert_allclose(ours[f"rrs_{sensor}_{band}"], ref[band], rtol=0, atol=1e-9)
        assert np.allclose(ours[f"coverage_{sensor}_{band}"], 1.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("sensor", MSI)
def test_real_spectra_match_production(sensor, real, tmp_path):
    x, spectra, meta, reference = real
    ours = resample(x, spectra, sensor)
    ref = production(x, spectra, sensor, tmp_path)
    stored = reference[reference["platform"] == sensor].reset_index(drop=True)
    assert list(stored["measurement"]) == list(meta["measurement"])
    for band in PRIMARY:
        np.testing.assert_allclose(ours[f"rrs_{sensor}_{band}"], ref[band], rtol=0, atol=1e-9)
        # values written by the production run (S2A B1: the production SRF file had 8 decimals)
        np.testing.assert_allclose(ours[f"rrs_{sensor}_{band}"], stored[band], rtol=0, atol=1e-9)


@pytest.mark.parametrize("sensor", MSI)
def test_b8_at_the_900_nm_end(sensor, real, tmp_path):
    x, spectra, _, _ = real
    ours = resample(x, spectra, sensor)
    ref = production(x, spectra, sensor, tmp_path)
    coverage = ours[f"coverage_{sensor}_B8"].to_numpy()
    assert np.allclose(coverage, B8_COVERAGE[sensor], atol=1e-5)
    assert np.isfinite(ours[f"rrs_{sensor}_B8"]).all()  # above the default limit of 0.99
    assert resample(x, spectra, sensor, min_coverage=0.995)[f"rrs_{sensor}_B8"].isna().all()
    # Production repeated Rrs(900) up to 2500 nm, which mixes Rrs(900) into B8 with a fixed weight f
    srf = load_srf(sensor)
    s = srf[srf["band"] == "B8"].set_index("wavelength_nm")["response"]
    s.index = s.index.astype(float)
    w = np.full(x.size, 1.0)
    w[0] = w[-1] = 0.5
    d_ours = np.sum(w * s.reindex(x).to_numpy())
    extra = 0.5 * s[900] + s[s.index > 900].sum()
    f = extra / (d_ours + extra)
    b8 = ours[f"rrs_{sensor}_B8"].to_numpy()
    np.testing.assert_allclose(ref["B8"].to_numpy() - b8, f * (spectra[:, -1] - b8), rtol=0, atol=1e-12)
    assert 0.005 < f < 0.01
    # B9, B11, B12 were Rrs(900) in production, here they are NaN without coverage
    for band in ("B9", "B11", "B12"):
        np.testing.assert_allclose(ref[band], spectra[:, -1], rtol=0, atol=1e-12)
        assert ours[f"rrs_{sensor}_{band}"].isna().all()
        assert (ours[f"coverage_{sensor}_{band}"] == 0).all()


def test_flat_nir_b8_equals_production(tmp_path):
    x, spectra = synthetic(n=4)
    spectra[:, x >= 760] = spectra[:, [410]]  # constant from 760 nm, so extrapolation changes nothing
    for sensor in MSI:
        ours = resample(x, spectra, sensor)
        ref = production(x, spectra, sensor, tmp_path)
        np.testing.assert_allclose(ours[f"rrs_{sensor}_B8"], ref["B8"], rtol=0, atol=1e-12)


def test_linear_spectrum_gives_value_at_band_centre():
    x = np.arange(350.0, 901.0)
    spectrum = 0.001 + 2e-5 * (x - 350.0)
    ours = resample(x, spectrum, MSI)
    for sensor in MSI:
        centres = band_centres(sensor)
        for band in PRIMARY:
            expected = 0.001 + 2e-5 * (centres[band] - 350.0)
            assert ours[f"rrs_{sensor}_{band}"].iloc[0] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("sensor", ["S3A", "S3B"])
def test_olci_matches_fine_grid_integral(sensor, real):
    """Narrow OLCI bands: union grid within 5e-6 sr-1 of SRF times the interpolated spectrum at 0.005 nm."""
    x, spectra, _, _ = real
    ours = resample(x, spectra[:4], sensor)
    srf = load_srf(sensor)
    for band in [f"Oa{i:02d}" for i in range(1, 19)]:
        part = srf[srf["band"] == band]
        xs, ys = part["wavelength_nm"].to_numpy(float), part["response"].to_numpy(float)
        fine = np.arange(xs[0], xs[-1], 0.005)
        s = np.interp(fine, xs, ys)
        for i in range(4):
            r = np.interp(fine, x, spectra[i])
            expected = trapezoid(s * r, fine) / trapezoid(s, fine)
            assert ours[f"rrs_{sensor}_{band}"].iloc[i] == pytest.approx(expected, rel=0, abs=5e-6), band


@pytest.mark.parametrize("sensor", ["S3A", "S3B"])
def test_olci_bands_above_900_nm_are_nan(sensor, real):
    x, spectra, _, _ = real
    ours = resample(x, spectra, sensor)
    inside = [f"Oa{i:02d}" for i in range(1, 19)]
    for band in inside:
        assert np.isfinite(ours[f"rrs_{sensor}_{band}"]).all(), band
        assert np.allclose(ours[f"coverage_{sensor}_{band}"], 1.0, rtol=0, atol=1e-12), band
    for band in ("Oa19", "Oa20", "Oa21"):
        assert ours[f"rrs_{sensor}_{band}"].isna().all(), band
    cov19 = ours[f"coverage_{sensor}_Oa19"].to_numpy()
    assert np.all((cov19 > 0.5) & (cov19 < 0.6))
    assert (ours[f"coverage_{sensor}_Oa20"] == 0).all() and (ours[f"coverage_{sensor}_Oa21"] == 0).all()
    # without the coverage limit Oa19 is the mean over the measured part, Oa20 and Oa21 stay NaN
    loose = resample(x, spectra, sensor, min_coverage=0.0)
    assert np.isfinite(loose[f"rrs_{sensor}_Oa19"]).all()
    assert loose[f"rrs_{sensor}_Oa20"].isna().all() and loose[f"rrs_{sensor}_Oa21"].isna().all()


def test_coverage_values():
    x = np.arange(350.0, 901.0)
    spectrum = np.full(x.size, 0.005)
    srf = load_srf("S2A")
    b4 = srf[srf["band"] == "B4"]
    full = trapezoid(b4["response"], b4["wavelength_nm"])
    cut = 665.0
    part = b4[b4["wavelength_nm"] <= cut]
    expected = trapezoid(part["response"], part["wavelength_nm"]) / full
    gap = spectrum.copy()
    gap[x > cut] = np.nan
    out = resample(x, np.vstack([spectrum, gap, np.full(x.size, np.nan)]), "S2A")
    assert out["coverage_S2A_B4"].iloc[0] == pytest.approx(1.0, abs=1e-12)
    assert out["coverage_S2A_B4"].iloc[1] == pytest.approx(expected, abs=1e-12)
    assert 0.3 < expected < 0.7
    assert np.isnan(out["rrs_S2A_B4"].iloc[1])
    assert out["rrs_S2A_B2"].iloc[1] == pytest.approx(0.005, abs=1e-15)
    # an all-NaN spectrum has no coverage and no values
    assert out.filter(like="coverage_").iloc[2].eq(0).all()
    assert out.filter(like="rrs_").iloc[2].isna().all()
    # one missing sample inside a band removes the two intervals next to it
    hole = spectrum.copy()
    hole[x == 560.0] = np.nan
    b3 = resample(x, hole, "S2A")["coverage_S2A_B3"].iloc[0]
    b3_srf = srf[srf["band"] == "B3"].set_index("wavelength_nm")["response"]
    lost = (b3_srf[559] + 2 * b3_srf[560] + b3_srf[561]) / 2 / trapezoid(b3_srf.to_numpy(), b3_srf.index)
    assert b3 == pytest.approx(1.0 - lost, abs=1e-12)
    # a grid starting at 400 nm covers only part of Oa01 (387.7-411.3 nm)
    x2 = np.arange(400.0, 901.0)
    olci = resample(x2, np.full(x2.size, 0.005), "S3A")
    assert 0.0 < olci["coverage_S3A_Oa01"].iloc[0] < 0.99 and np.isnan(olci["rrs_S3A_Oa01"].iloc[0])
    assert olci["rrs_S3A_Oa02"].iloc[0] == pytest.approx(0.005, abs=1e-15)


def test_output_layout():
    x, spectra = synthetic(n=3)
    frame = pd.DataFrame(spectra, index=pd.Index(["a", "b", "c"], name="id"))
    out = resample(x, frame)
    assert list(out.index) == ["a", "b", "c"]
    names = [c for c in out.columns if c.startswith("rrs_")]
    assert names[0] == "rrs_S2A_B1" and "rrs_S2A_B8A" in names and "rrs_S3B_Oa21" in names
    assert len(names) == 3 * 13 + 2 * 21
    assert len(out.columns) == 2 * len(names)
    single = resample(x, spectra[0], "S3A")
    assert single.shape == (1, 42) and single.columns[0] == "rrs_S3A_Oa01"
    with pytest.raises(ValueError):
        resample(x[::-1], spectra, "S2A")
    with pytest.raises(ValueError):
        resample(x, spectra[:, :-1], "S2A")
    with pytest.raises(ValueError):
        resample(x, spectra, "S9Z")


def test_extra_srf_file_is_picked_up(tmp_path):
    data = tmp_path / "srf_data"
    data.mkdir()
    for path in DATA_DIR.glob("*.csv"):
        shutil.copy(path, data / path.name)
    assert "S3C" not in available_sensors(data)
    shutil.copy(DATA_DIR / "olci_s3b.csv", data / "olci_s3c.csv")
    # a user table in the ESA layout (semicolons, SR_WL, S2x_SR_AV_Bn)
    s2a = load_srf("S2A").pivot(index="wavelength_nm", columns="band", values="response")[["B2", "B3"]]
    s2a.index = s2a.index.astype(int)
    s2a.index.name = "SR_WL"
    s2a.columns = ["S2A_SR_AV_B2", "S2A_SR_AV_B3"]
    s2a.to_csv(data / "user_test1.csv", sep=";")
    assert available_sensors(data) == ["S2A", "S2B", "S2C", "S3A", "S3B", "S3C", "TEST1"]

    x, spectra = synthetic(n=5)
    out = resample(x, spectra, ["S3B", "S3C", "TEST1"], data_dir=data)
    for i in range(1, 22):
        band = f"Oa{i:02d}"
        np.testing.assert_allclose(out[f"rrs_S3C_{band}"], out[f"rrs_S3B_{band}"], rtol=0, atol=1e-15)
    assert list(out.filter(like="rrs_TEST1").columns) == ["rrs_TEST1_B2", "rrs_TEST1_B3"]
    np.testing.assert_allclose(out["rrs_TEST1_B3"], resample(x, spectra, "S2A")["rrs_S2A_B3"], rtol=0, atol=1e-15)
    # a file path works too, and band centres fall back to the first moment of the SRF
    direct = resample(x, spectra, data / "user_test1.csv")
    np.testing.assert_allclose(direct["rrs_TEST1_B2"], out["rrs_TEST1_B2"], rtol=0, atol=1e-15)
    assert band_centres("S3C", data_dir=data)["Oa17"] == pytest.approx(band_centres("S3B")["Oa17"], abs=1e-3)
    assert band_centres("TEST1", data_dir=data)["B3"] == pytest.approx(band_centres("S2A")["B3"], abs=1e-9)
