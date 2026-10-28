"""WISPstation sites and the detection of the station from L1 file names."""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

DATALAKES = "datalakes"
WISPCLOUD = "wispcloud"


@dataclass(frozen=True)
class Station:
    """One WISPstation site.

    ``timezone`` is the local time zone used for calendar days and solar noon in the
    illumination check. ``view_zenith_deg`` and ``relative_azimuth_deg`` describe the sensor
    geometry. ``file_format`` is ``"datalakes"`` or ``"wispcloud"`` (None when unknown).
    """

    code: str
    name: str
    lat: float
    lon: float
    altitude_m: float
    timezone: str
    view_zenith_deg: float = 45.0
    relative_azimuth_deg: float = 135.0
    file_format: str | None = None

    @property
    def surface_pressure_hpa(self) -> float:
        """Pressure from the station altitude with the standard barometric formula."""
        return barometric_pressure_hpa(self.altitude_m)


def barometric_pressure_hpa(altitude_m: float) -> float:
    """Standard atmosphere pressure (hPa) at ``altitude_m`` metres."""
    return 1013.25 * (1.0 - 2.25577e-5 * float(altitude_m)) ** 5.25588


STATIONS: dict[str, Station] = {
    "GFN": Station("GFN", "Greifensee", 47.3665, 8.6648333, 435.0, "Europe/Zurich",
                   file_format=DATALAKES),
    "LXP": Station("LXP", "Lake Geneva (LéXPLORE)", 46.5002275, 6.6608353, 372.0, "Europe/Zurich",
                   file_format=DATALAKES),
    "TRS012": Station("TRS012", "Lake Trasimeno WISPstation012", 43.123064, 12.129649, 258.0,
                      "Europe/Rome", file_format=WISPCLOUD),
    "TRS001": Station("TRS001", "Lake Trasimeno WISPstation001", 43.123064, 12.129649, 258.0,
                      "Europe/Rome", file_format=WISPCLOUD),
}

_ALIASES = {
    "GREIFENSEE": "GFN", "GREIFEN": "GFN", "LAKE_GREIFEN": "GFN",
    "LEXPLORE": "LXP", "LAKE GENEVA": "LXP", "GENEVA": "LXP",
    "WISPSTATION012": "TRS012", "WISPSTATION001": "TRS001",
}

# File name patterns: Datalakes files of both spellings and WISPcloud spectral files.
_PATTERNS = [
    (re.compile(r"^L1B_Lake_Greifen_Sp(?:ectral|rectral)_(\d{8})_(\d{6})", re.IGNORECASE), "GFN", DATALAKES),
    (re.compile(r"^L1_Lexplore_Sp(?:ectral|rectral)_(\d{8})_(\d{6})", re.IGNORECASE), "LXP", DATALAKES),
    (re.compile(r"^(WISPstation\d{3})_(\d{8})_L1_SPEC", re.IGNORECASE), None, WISPCLOUD),
]


def get_station(station) -> Station:
    """Return the :class:`Station` for a Station, a built-in code or a known alias."""
    if isinstance(station, Station):
        return station
    key = str(station).strip().upper()
    key = _ALIASES.get(key, key)
    if key not in STATIONS:
        raise KeyError(f"Unknown station {station!r}. Built-in stations: {', '.join(STATIONS)}. "
                       "Use custom_station() for another site.")
    return STATIONS[key]


def custom_station(code: str, lat: float, lon: float, altitude_m: float, name: str | None = None,
                   timezone: str = "UTC", file_format: str | None = None,
                   view_zenith_deg: float = 45.0, relative_azimuth_deg: float = 135.0) -> Station:
    """A user-defined station.

    Give the local ``timezone`` (for example ``"Europe/Berlin"``) when the illumination check is
    used: calendar days and solar noon follow it.
    """
    return Station(str(code), name or str(code), float(lat), float(lon), float(altitude_m), timezone,
                   view_zenith_deg=float(view_zenith_deg),
                   relative_azimuth_deg=float(relative_azimuth_deg), file_format=file_format)


def parse_l1_filename(path) -> tuple[str | None, str | None, str | None]:
    """Return (station code, file format, date YYYYMMDD) from an L1 file name.

    The code is None for a WISPcloud station number without a built-in station, in which case
    the second element is still ``"wispcloud"``. All three are None for an unknown name.
    """
    name = Path(str(path)).name
    for pattern, code, fmt in _PATTERNS:
        m = pattern.match(name)
        if not m:
            continue
        if fmt == WISPCLOUD:
            key = _ALIASES.get(m.group(1).upper())
            return key, fmt, m.group(2)
        return code, fmt, m.group(1)
    return None, None, None


def detect_station(path) -> Station | None:
    """The built-in station of an L1 file name, or None when the name is not recognised."""
    code, _, _ = parse_l1_filename(path)
    return STATIONS[code] if code else None


def wispcloud_station_id(path) -> str | None:
    """The WISPcloud station identifier (for example ``WISPstation012``) in a file name."""
    m = _PATTERNS[2][0].match(Path(str(path)).name)
    return m.group(1) if m else None


def with_format(station: Station, file_format: str) -> Station:
    """Copy of ``station`` with ``file_format`` set."""
    return replace(station, file_format=file_format)
