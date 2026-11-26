"""Automatic phase correction (ACME: Automated phase Correction based on Minimization of Entropy).

Reference: L. Chen et al., J. Magn. Reson. 158, 164-168 (2002).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from nmrproc.data import Spectrum
from nmrproc.processing import phase_correct


def _acme_cost(spec, p0, p1, pivot, gamma, p1_weight):
    r = phase_correct(spec, p0, p1, pivot).real
    d = np.abs(np.diff(r))
    total = d.sum()
    if total == 0:
        return 0.0
    h = d / total
    h = h[h > 0]
    entropy = -np.sum(h * np.log(h))
    neg = r[r < 0]
    penalty = gamma * np.sum(neg**2) / (np.sum(r**2) + 1e-30)
    # Weak preference for small first-order phase: with few peaks p1 is poorly determined.
    return entropy + penalty + p1_weight * (p1 / 360.0) ** 2


def auto_phase(spectrum: Spectrum, gamma=20.0, p1_weight=0.5, p1_max=720.0, fit_p1=True):
    """Find the additional zero- and first-order phase (degrees) that best phases the spectrum.

    Minimises the entropy of the spectrum's derivative plus a penalty for negative intensity.
    The search first scans zero-order phase alone, then refines both terms, with the first-order
    phase pivoted on the tallest peak (well conditioned) and kept within +/- ``p1_max``.
    Set ``fit_p1=False`` to fit the zero-order phase only.

    Automatic phasing is a good starting point, typically within a few degrees but occasionally
    10 degrees or more off when the first-order error is large or few peaks are present; check
    the result and refine it by hand if needed.

    The returned values refer to the left edge of the spectrum (TopSpin convention) and can be
    passed straight to ``phase_spectrum`` or ``phase_correct``. Returns (p0, p1).
    """
    spec = np.asarray(spectrum.data, dtype=complex)
    n = len(spec)
    pivot = float(np.argmax(np.abs(spec))) / n

    # Stage 1: coarse scan of p0 with p1 = 0, then polish p0 alone.
    grid = np.arange(0.0, 360.0, 10.0)
    costs = [_acme_cost(spec, p, 0.0, pivot, gamma, p1_weight) for p in grid]
    start = float(grid[int(np.argmin(costs))])
    res = minimize(lambda v: _acme_cost(spec, v[0], 0.0, pivot, gamma, p1_weight), [start],
                   method="Nelder-Mead", options={"xatol": 0.01, "fatol": 1e-10})
    p0_pivot, p1 = float(res.x[0]), 0.0

    # Stage 2: refine p0 and p1 together.
    if fit_p1:
        res = minimize(lambda v: _acme_cost(spec, v[0], v[1], pivot, gamma, p1_weight),
                       [p0_pivot, 0.0], method="Nelder-Mead",
                       bounds=[(None, None), (-p1_max, p1_max)],
                       options={"xatol": 0.01, "fatol": 1e-10, "maxiter": 800})
        p0_pivot, p1 = float(res.x[0]), float(res.x[1])

    # Convert from "pivot at tallest peak" to "pivot at left edge"; wrap p0 into (-180, 180].
    p0 = p0_pivot - p1 * pivot
    p0 = (p0 + 180.0) % 360.0 - 180.0
    return p0, p1
