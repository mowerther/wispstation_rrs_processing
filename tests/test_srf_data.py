"""SRF tables shipped with the package."""
import hashlib
import json
import re

import numpy as np
import pytest

from wispstation_rrs.srf import DATA_DIR, DEFAULT_SENSORS, available_sensors, band_centres, load_srf

trapezoid = getattr(np, "trapezoid", None) or np.trapz

MSI_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
OLCI_BANDS = [f"Oa{i:02d}" for i in range(1, 22)]


def test_available_sensors():
    assert available_sensors() == list(DEFAULT_SENSORS)


def test_manifest_checksums():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    tables = sorted(p.name for p in DATA_DIR.glob("*.csv"))
    assert sorted(manifest) == tables
    for name, entry in manifest.items():
        digest = hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
        assert digest == entry["sha256"], name
    for name in ("msi_s2a.csv", "msi_s2b.csv", "msi_s2c.csv"):
        assert manifest[name]["release"] == "4.0"
        assert manifest[name]["document"] == "COPE-GSEG-EOPG-TN-15-0007"


def test_no_local_paths_in_data_folder():
    pattern = re.compile(r"\b[A-Za-z]:[\\/]|[\\/]Users[\\/]|[\\/]home[\\/]")
    for path in list(DATA_DIR.glob("*.json")) + list(DATA_DIR.glob("*.md")):
        assert not pattern.search(path.read_text(encoding="utf-8")), path.name


@pytest.mark.parametrize("sensor", ["S2A", "S2B", "S2C"])
def test_msi_tables(sensor):
    table = load_srf(sensor)
    assert table.attrs["sensor"] == sensor
    assert list(dict.fromkeys(table["band"])) == MSI_BANDS
    for _, part in table.groupby("band", sort=False):
        wl = part["wavelength_nm"].to_numpy()
        assert wl[0] == 300 and wl[-1] == 2600 and np.all(np.diff(wl) == 1)
        assert part["response"].min() >= 0 and part["response"].max() == pytest.approx(1.0, abs=1e-6)
    # the band table centre is the first moment of the response (ESA equivalent wavelength)
    centres = band_centres(sensor)
    for band, part in table.groupby("band", sort=False):
        x, y = part["wavelength_nm"].to_numpy(float), part["response"].to_numpy(float)
        assert trapezoid(x * y, x) / trapezoid(y, x) == pytest.approx(centres[band], abs=1e-6)
    assert band_centres(sensor, nominal=True)["B8A"] == 865.0


@pytest.mark.parametrize("sensor", ["S3A", "S3B"])
def test_olci_tables(sensor):
    table = load_srf(sensor)
    assert list(dict.fromkeys(table["band"])) == OLCI_BANDS
    assert (table.groupby("band").size() == 200).all()
    centres = band_centres(sensor)
    nominal = band_centres(sensor, nominal=True)
    assert list(centres.index) == OLCI_BANDS
    assert nominal["Oa17"] == 865.0 and nominal["Oa21"] == 1020.0
    for band, part in table.groupby("band", sort=False):
        x, y = part["wavelength_nm"].to_numpy(float), part["response"].to_numpy(float)
        assert trapezoid(x * y, x) / trapezoid(y, x) == pytest.approx(centres[band], abs=1e-3)
        assert abs(centres[band] - nominal[band]) < 5.0
