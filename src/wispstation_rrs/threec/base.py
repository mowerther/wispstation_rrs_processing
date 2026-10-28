"""One interface for the 3C implementations.

``get_model(name, **options)`` returns an object with ``.name`` and
``.fit(wavelength, ed, ls, lu, sza, vza, raa, rh, pressure, water_temp)``, which fits one
measurement and returns a :class:`~wispstation_rrs.types.ThreeCResult` on the 350-900 nm,
1 nm model grid. The inputs are 1-D spectra on any wavelength grid that covers 350-900 nm.
Inputs that an implementation does not use are accepted and ignored.

Names
    ``jax``  3C of Groetsch et al. (2017), JAX implementation of production (Werther).
    ``o25``  3C-O25 of Pitarch (2026), vendored upstream version 1.0.4.

For a process pool, pass ``init_worker`` with ``(name, options)`` as initialiser and map
``fit_one`` over argument tuples or dictionaries, so that every process builds (and for JAX
compiles) its model once.
"""
from __future__ import annotations

import importlib

import numpy as np

from ..types import ThreeCResult

# model grid shared by both implementations (production grid of the JAX runs)
MODEL_GRID = np.arange(350, 901, dtype=np.int64)

_MODELS = {
    "jax": ("wispstation_rrs.threec.jax_werther.adapter", "JaxWertherModel"),
    "o25": ("wispstation_rrs.threec.o25_pitarch.adapter", "O25PitarchModel"),
}


def available_models() -> list[str]:
    """Names accepted by :func:`get_model`."""
    return sorted(_MODELS)


def get_model(name: str, **options):
    """Return the 3C implementation ``name`` ('jax' or 'o25') built with ``options``."""
    key = str(name).lower()
    if key not in _MODELS:
        raise ValueError(f"unknown 3C model {name!r}, expected one of {available_models()}")
    module, cls = _MODELS[key]
    return getattr(importlib.import_module(module), cls)(**options)


def finite_or_none(value):
    """Return ``value`` as float, or None when it is None or not finite."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def to_model_grid(wavelength, grid, *spectra):
    """Interpolate spectra linearly (np.interp, as production) from ``wavelength`` to ``grid``.

    The input grid must cover the model grid. A decreasing input grid is sorted first.
    """
    wl = np.asarray(wavelength, dtype=float).ravel()
    arrays = [np.asarray(s, dtype=float).ravel() for s in spectra]
    for a in arrays:
        if a.shape != wl.shape:
            raise ValueError(f"spectrum has {a.size} values, wavelength has {wl.size}")
    if wl.size < 2 or not np.all(np.isfinite(wl)):
        raise ValueError("wavelength needs at least two finite values")
    if np.any(np.diff(wl) <= 0):
        order = np.argsort(wl, kind="stable")
        wl = wl[order]
        arrays = [a[order] for a in arrays]
        if np.any(np.diff(wl) == 0):
            raise ValueError("wavelength has duplicate values")
    if wl[0] > grid[0] or wl[-1] < grid[-1]:
        raise ValueError(f"input wavelengths {wl[0]:g}-{wl[-1]:g} nm do not cover {grid[0]}-{grid[-1]} nm")
    return [np.interp(grid, wl, a) for a in arrays]


def failed_result(message: str, grid=MODEL_GRID) -> ThreeCResult:
    """Result of a fit that raised: NaN spectra, success False and the error message."""
    nan = np.full(len(grid), np.nan)
    return ThreeCResult(wavelength=np.asarray(grid, dtype=float), rrs=nan.copy(), rrs_model=nan.copy(),
                        rrs_surface=nan.copy(), params={}, cost=float("nan"), success=False,
                        message=message, nfev=0)


# model of this process, set by init_worker
_worker_model = None


def init_worker(name: str, options: dict | None = None) -> None:
    """Process pool initialiser: build the model once in this process."""
    global _worker_model
    _worker_model = get_model(name, **(options or {}))


def fit_one(args) -> ThreeCResult:
    """Fit one measurement with the model of this process.

    ``args`` is a tuple in the order of ``fit`` or a dictionary of its keyword arguments.
    A fit that raises gives a result with NaN spectra, ``success`` False and the error in
    ``message``, as the production scripts recorded failed measurements.
    """
    if _worker_model is None:
        raise RuntimeError("init_worker was not called in this process")
    try:
        if isinstance(args, dict):
            return _worker_model.fit(**args)
        return _worker_model.fit(*args)
    except Exception as exc:  # noqa: BLE001
        return failed_result(f"{type(exc).__name__}: {exc}", getattr(_worker_model, "grid", MODEL_GRID))
