"""Quality Water Index Polynomial (QWIP) of hyperspectral Rrs (Dierssen et al., 2022).

AVW (apparent visible wavelength) = sum(Rrs) / sum(Rrs / wavelength) over 400-700 nm.
NDI = (Rrs(665) - Rrs(492)) / (Rrs(665) + Rrs(492)) at the bands nearest 492 and 665 nm
(within 2 nm). QWIP = polynomial(AVW) - NDI, and a spectrum passes when ``|QWIP| <= 0.2``.
In the processing QWIP is computed on the 3C Rrs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Polynomial of the paper (highest power first); keep these values.
QWIP_COEFFICIENTS = (-8.399884740300151e-09, 1.715532100780679e-05, -1.301670056641901e-02,
                     4.357837742180596e+00, -5.449532021524279e+02)


def _nearest(wavelength: np.ndarray, target: float, tol: float) -> int | None:
    diffs = np.abs(wavelength - target)
    i = int(np.argmin(diffs))
    return i if diffs[i] <= tol else None


def qwip(wavelength, rrs, threshold: float = 0.2, avw_range=(400.0, 700.0), wl_tol: float = 2.0,
         coefficients=QWIP_COEFFICIENTS) -> pd.DataFrame:
    """QWIP of each spectrum.

    ``rrs`` has the shape (n, m) (or (m,) for one spectrum) on ``wavelength`` (m,) in nm.
    Returns a DataFrame with ``avw_nm``, ``ndi``, ``qwip_score`` and ``qwip_ok`` (False when the
    score is NaN). NaN values of Rrs are left out of the AVW sums.
    """
    wl = np.asarray(wavelength, dtype=float).ravel()
    r = np.atleast_2d(np.asarray(rrs, dtype=float))
    if r.shape[1] != wl.size:
        raise ValueError(f"rrs has {r.shape[1]} bands, wavelength has {wl.size}")

    vis = (wl >= avw_range[0]) & (wl <= avw_range[1])
    rv = r[:, vis]
    wv = wl[vis]
    valid = np.isfinite(rv)
    sum_rrs = np.where(valid, rv, 0.0).sum(axis=1)
    sum_rrs_div_wave = np.where(valid, rv / wv, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        avw = sum_rrs / sum_rrs_div_wave
    avw[~valid.any(axis=1) | (np.abs(sum_rrs_div_wave) <= 1e-12)] = np.nan

    i492 = _nearest(wl, 492.0, wl_tol)
    i665 = _nearest(wl, 665.0, wl_tol)
    if i492 is None or i665 is None:
        ndi = np.full(r.shape[0], np.nan)
    else:
        num = r[:, i665] - r[:, i492]
        den = r[:, i665] + r[:, i492]
        with np.errstate(invalid="ignore", divide="ignore"):
            ndi = np.where(np.abs(den) < 1e-12, np.nan, num / den)

    p = coefficients
    ndi_pred = (p[0] * avw ** 4 + p[1] * avw ** 3 + p[2] * avw ** 2 + p[3] * avw ** 1 + p[4])
    score = ndi_pred - ndi
    ok = np.isfinite(score) & (np.abs(score) <= float(threshold))
    return pd.DataFrame({"avw_nm": avw, "ndi": ndi, "qwip_score": score, "qwip_ok": ok})
