"""Data containers shared by the readers, the 3C fits, the checks and the output tables."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Spectra:
    """Radiometry of one station, one row per measurement time.

    ``meta`` holds one row per measurement with at least the columns ``station``, ``file`` and
    ``time_utc`` (timezone-aware UTC timestamps) plus the quality flags of the L1 file
    (``ed_qual_ok``, ``ls_qual_ok``, ``lu_qual_ok``). ``ed``, ``ls`` and ``lu`` have the shape
    (number of measurements, number of wavelengths), in W m-2 nm-1 (Ed) and W m-2 nm-1 sr-1 (Ls, Lu).
    ``ls`` is the sky radiance (Ld in the Datalakes files).
    """

    station: str
    wavelength: np.ndarray
    ed: np.ndarray
    ls: np.ndarray
    lu: np.ndarray
    meta: pd.DataFrame
    attrs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n, m = len(self.meta), len(self.wavelength)
        for name in ("ed", "ls", "lu"):
            arr = getattr(self, name)
            if arr.shape != (n, m):
                raise ValueError(f"{name} has shape {arr.shape}, expected {(n, m)}")

    def __len__(self) -> int:
        return len(self.meta)

    def subset(self, index) -> "Spectra":
        """Return the measurements at the positional ``index`` (array of int or bool)."""
        index = np.asarray(index)
        return Spectra(self.station, self.wavelength, self.ed[index], self.ls[index], self.lu[index],
                       self.meta.iloc[index].reset_index(drop=True), dict(self.attrs))


@dataclass
class ThreeCResult:
    """Result of one 3C fit.

    ``rrs`` is the retrieved Rrs (Lu/Ed minus the modelled surface reflection) on ``wavelength``
    in sr-1. ``rrs_model`` is the modelled water-leaving part and ``rrs_surface`` the modelled
    surface reflection, both on ``wavelength``. ``params`` holds the final values of all model
    parameters. ``cost`` is the weighted sum of squared residuals at the solution.
    """

    wavelength: np.ndarray
    rrs: np.ndarray
    rrs_model: np.ndarray
    rrs_surface: np.ndarray
    params: dict
    cost: float
    success: bool
    message: str = ""
    nfev: int = 0
