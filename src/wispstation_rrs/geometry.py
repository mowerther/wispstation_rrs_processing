"""Sun position for the measurements and the daily solar schedule of the illumination check.

Sun position: :func:`solar_angles` uses ``pvlib.solarposition.get_solarposition`` (NREL SPA,
``method="nrel_numpy"``). The solar zenith angle ``sza`` is pvlib's ``apparent_zenith``, the
zenith corrected for atmospheric refraction, computed with sea-level pressure (101325 Pa) and
15 degC. It agrees with the solar zenith angles of the paper processing within 0.006 degrees,
closer than the true zenith. ``sza_true`` is pvlib's
``zenith`` and ``saa`` is pvlib's ``azimuth`` (degrees clockwise from north).

Solar schedule: :func:`solar_schedule` reproduces the sunrise, solar noon and sunset times that
the illumination criterion of the paper processing used: the NOAA equations (as in astral 1.x), sunrise and sunset for a solar zenith of 90.833 degrees, times in
UTC truncated to whole seconds.
"""
from __future__ import annotations

import datetime as _dt
from math import acos, asin, cos, degrees, floor, radians, sin, tan

import numpy as np
import pandas as pd

SUNRISE_ZENITH_DEG = 90.833
_RISING, _SETTING = 1, -1


def _as_utc_index(times) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    return idx.tz_convert("UTC")


def solar_angles(times, lat: float, lon: float, altitude_m: float = 0.0,
                 pressure_pa: float = 101325.0, temperature_c: float = 15.0) -> pd.DataFrame:
    """Solar zenith and azimuth angles (degrees) at ``times`` for one location.

    ``times`` are timezone-aware timestamps (naive values are taken as UTC). Returns a DataFrame
    with one row per time and the columns ``time_utc``, ``sza`` (apparent zenith), ``sza_true``
    (zenith without refraction) and ``saa`` (azimuth). ``altitude_m`` only enters the parallax
    correction: refraction uses ``pressure_pa`` and ``temperature_c``.
    """
    from pvlib.solarposition import get_solarposition

    idx = _as_utc_index(times)
    if len(idx) == 0:
        return pd.DataFrame({"time_utc": idx, "sza": [], "sza_true": [], "saa": []})
    sp = get_solarposition(idx, lat, lon, altitude=altitude_m, pressure=pressure_pa,
                           method="nrel_numpy", temperature=temperature_c)
    return pd.DataFrame({
        "time_utc": idx,
        "sza": sp["apparent_zenith"].to_numpy(dtype=float),
        "sza_true": sp["zenith"].to_numpy(dtype=float),
        "saa": sp["azimuth"].to_numpy(dtype=float),
    })


def solar_angles_for_station(station, times) -> pd.DataFrame:
    """:func:`solar_angles` for a :class:`~wispstation_rrs.stations.Station` (or its code)."""
    from .stations import get_station

    st = get_station(station)
    return solar_angles(times, st.lat, st.lon, altitude_m=0.0)


# NOAA solar equations of the solar schedule of the paper processing.

def _julianday(day: _dt.date) -> float:
    year, month = day.year, day.month
    if month <= 2:
        year -= 1
        month += 12
    a = floor(year / 100.0)
    b = 2 - a + floor(a / 4.0)
    jd = floor(365.25 * (year + 4716)) + floor(30.6001 * (month + 1)) + day.day - 1524.5
    if jd > 2299160.4999999:
        jd += b
    return jd


def _jday_to_jcentury(julianday: float) -> float:
    return (julianday - 2451545.0) / 36525.0


def _jcentury_to_jday(juliancentury: float) -> float:
    return (juliancentury * 36525.0) + 2451545.0


def _geom_mean_long_sun(t: float) -> float:
    l0 = 280.46646 + t * (36000.76983 + 0.0003032 * t)
    return l0 % 360.0


def _geom_mean_anomaly_sun(t: float) -> float:
    return 357.52911 + t * (35999.05029 - 0.0001537 * t)


def _eccentricity_earth_orbit(t: float) -> float:
    return 0.016708634 - t * (0.000042037 + 0.0000001267 * t)


def _sun_eq_of_center(t: float) -> float:
    m = _geom_mean_anomaly_sun(t)
    mrad = radians(m)
    sinm = sin(mrad)
    sin2m = sin(mrad + mrad)
    sin3m = sin(mrad + mrad + mrad)
    return (sinm * (1.914602 - t * (0.004817 + 0.000014 * t))
            + sin2m * (0.019993 - 0.000101 * t) + sin3m * 0.000289)


def _sun_true_long(t: float) -> float:
    return _geom_mean_long_sun(t) + _sun_eq_of_center(t)


def _sun_apparent_long(t: float) -> float:
    omega = 125.04 - 1934.136 * t
    return _sun_true_long(t) - 0.00569 - 0.00478 * sin(radians(omega))


def _mean_obliquity_of_ecliptic(t: float) -> float:
    seconds = 21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))
    return 23.0 + (26.0 + (seconds / 60.0)) / 60.0


def _obliquity_correction(t: float) -> float:
    omega = 125.04 - 1934.136 * t
    return _mean_obliquity_of_ecliptic(t) + 0.00256 * cos(radians(omega))


def _sun_declination(t: float) -> float:
    e = _obliquity_correction(t)
    lambd = _sun_apparent_long(t)
    return degrees(asin(sin(radians(e)) * sin(radians(lambd))))


def _var_y(t: float) -> float:
    y = tan(radians(_obliquity_correction(t)) / 2.0)
    return y * y


def _eq_of_time(t: float) -> float:
    l0 = _geom_mean_long_sun(t)
    e = _eccentricity_earth_orbit(t)
    m = _geom_mean_anomaly_sun(t)
    y = _var_y(t)
    sin2l0 = sin(2.0 * radians(l0))
    sinm = sin(radians(m))
    cos2l0 = cos(2.0 * radians(l0))
    sin4l0 = sin(4.0 * radians(l0))
    sin2m = sin(2.0 * radians(m))
    etime = (y * sin2l0 - 2.0 * e * sinm + 4.0 * e * y * sinm * cos2l0
             - 0.5 * y * y * sin4l0 - 1.25 * e * e * sin2m)
    return degrees(etime) * 4.0


def _hour_angle(latitude: float, declination: float, zenith: float) -> float:
    lat_rad = radians(latitude)
    sd = radians(declination)
    eq = radians(zenith)
    h = (cos(eq) / (cos(lat_rad) * cos(sd))) - tan(lat_rad) * tan(sd)
    return acos(h)


def _to_datetime(day: _dt.date, time_utc_min: float) -> _dt.datetime:
    hours = time_utc_min / 60.0
    hour = int(hours)
    minute = int((hours - hour) * 60)
    second = int((((hours - hour) * 60) - minute) * 60)
    if second > 59:
        second -= 60
        minute += 1
    elif second < 0:
        second += 60
        minute -= 1
    if minute > 59:
        minute -= 60
        hour += 1
    elif minute < 0:
        minute += 60
        hour -= 1
    if hour > 23:
        hour -= 24
        day += _dt.timedelta(days=1)
    elif hour < 0:
        hour += 24
        day -= _dt.timedelta(days=1)
    return _dt.datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=_dt.timezone.utc)


def _calc_time(day: _dt.date, latitude: float, longitude: float, zenith: float,
               direction: int) -> _dt.datetime:
    latitude = min(max(latitude, -89.8), 89.8)
    t = _jday_to_jcentury(_julianday(day))
    eqtime = _eq_of_time(t)
    solar_dec = _sun_declination(t)
    # The first estimate uses the setting hour angle for both events, as the paper processing
    # schedule did: this reproduces its sunrise times to the second.
    hourangle = -_hour_angle(latitude, solar_dec, zenith)
    delta = -longitude - degrees(hourangle)
    time_utc = 720.0 + 4.0 * delta - eqtime

    newt = _jday_to_jcentury(_jcentury_to_jday(t) + time_utc / 1440.0)
    eqtime = _eq_of_time(newt)
    solar_dec = _sun_declination(newt)
    hourangle = _hour_angle(latitude, solar_dec, zenith)
    if direction == _SETTING:
        hourangle = -hourangle
    delta = -longitude - degrees(hourangle)
    time_utc = 720 + 4 * delta - eqtime
    return _to_datetime(day, time_utc)


def solar_noon_utc(day: _dt.date, longitude: float) -> _dt.datetime:
    """Solar noon (UTC, whole seconds) of the date ``day`` at ``longitude`` (degrees east)."""
    newt = _jday_to_jcentury(_julianday(day) + 0.5 + -longitude / 360.0)
    eqtime = _eq_of_time(newt)
    time_utc = 720.0 + (-longitude * 4.0) - eqtime
    return _to_datetime(day, time_utc)


def solar_schedule(day, lat: float, lon: float) -> dict:
    """Sunrise, solar noon and sunset (UTC, whole seconds) of the date ``day``.

    Returns a dict with the keys ``sunrise``, ``noon`` and ``sunset``. Sunrise and sunset are
    ``None`` when the sun does not cross the 90.833 degree zenith on that date.
    """
    day = pd.Timestamp(day).date()
    out = {"noon": solar_noon_utc(day, lon)}
    for key, direction in (("sunrise", _RISING), ("sunset", _SETTING)):
        try:
            out[key] = _calc_time(day, lat, lon, SUNRISE_ZENITH_DEG, direction)
        except ValueError:
            out[key] = None
    return out


def _seconds_of_day(ts) -> int:
    return ts.hour * 3600 + ts.minute * 60 + ts.second


def solar_schedule_table(dates, lat: float, lon: float, timezone: str) -> pd.DataFrame:
    """Daily solar schedule of the illumination criterion.

    ``dates`` are local calendar dates. Returns one row per date with ``date_local``,
    ``sunrise_utc``, ``noon_utc``, ``sunset_utc`` (timezone-aware UTC), ``sunrise_local``,
    ``noon_local``, ``sunset_local`` (naive local clock time), ``sunrise_sec`` and ``sunset_sec``
    (whole seconds after local midnight) and ``day_length_s`` (sunset_sec minus sunrise_sec).
    """
    rows = []
    for day in sorted({pd.Timestamp(d).date() for d in dates}):
        sched = solar_schedule(day, lat, lon)
        row = {"date_local": day}
        for key in ("sunrise", "noon", "sunset"):
            value = sched[key]
            utc = pd.Timestamp(value) if value is not None else pd.NaT
            row[f"{key}_utc"] = utc
            row[f"{key}_local"] = utc.tz_convert(timezone).tz_localize(None) if value is not None else pd.NaT
        if sched["sunrise"] is not None and sched["sunset"] is not None:
            row["sunrise_sec"] = _seconds_of_day(row["sunrise_local"])
            row["sunset_sec"] = _seconds_of_day(row["sunset_local"])
            row["day_length_s"] = row["sunset_sec"] - row["sunrise_sec"]
        else:
            row["sunrise_sec"] = row["sunset_sec"] = row["day_length_s"] = np.nan
        rows.append(row)
    cols = ["date_local", "sunrise_utc", "noon_utc", "sunset_utc", "sunrise_local", "noon_local",
            "sunset_local", "sunrise_sec", "sunset_sec", "day_length_s"]
    return pd.DataFrame(rows, columns=cols)
