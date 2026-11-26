"""Baseline correction by polynomial fitting to signal-free regions of the spectrum."""
from __future__ import annotations

import numpy as np
from numpy.polynomial import Chebyshev

from nmrproc.data import Spectrum
from nmrproc.processing import estimate_noise


def find_baseline_points(y, window=None, factor=3.0, margin=None):
    """Boolean mask marking points that contain only baseline (no NMR signal).

    A point is signal if the local roughness (standard deviation of point-to-point differences
    in a sliding window) exceeds ``factor`` times the noise level. Signal regions are widened
    by ``margin`` points so that peak tails are excluded.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    window = window or max(16, n // 512)
    margin = margin if margin is not None else window
    d = np.diff(y, prepend=y[0])
    noise = estimate_noise(y)
    # sliding RMS of the differences
    kernel = np.ones(window) / window
    local = np.sqrt(np.convolve(d**2, kernel, mode="same"))
    signal = local > factor * np.sqrt(2) * noise
    if margin > 0:
        signal = np.convolve(signal.astype(float), np.ones(2 * margin + 1), mode="same") > 0
    return ~signal


def baseline_noise(y, mask=None):
    """Effective noise level: robust spread of the intensity in the signal-free regions.

    Unlike ``estimate_noise`` (white noise only), this includes slow baseline wobble, so it is
    the right yardstick for deciding whether something is a real peak.
    """
    y = np.asarray(y, dtype=float)
    if mask is None:
        mask = find_baseline_points(y)
    v = y[mask]
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def fit_baseline(y, mask, order=5, n_iter=5, clip=3.0):
    """Fit a Chebyshev polynomial of the given order to y[mask]; return the baseline curve.

    Points further than ``clip`` standard deviations from the fit are dropped and the fit
    repeated, so that leftover signal tails do not pull the baseline.
    """
    y = np.asarray(y, dtype=float)
    x = np.linspace(-1.0, 1.0, len(y))
    use = mask.copy()
    if use.sum() <= order + 1:
        raise ValueError("Too few baseline points to fit; lower the order or the threshold")
    fit = Chebyshev.fit(x[use], y[use], order)
    for _ in range(n_iter):
        resid = y - fit(x)
        sigma = np.std(resid[use])
        keep = use & (np.abs(resid) < clip * sigma)
        if keep.sum() <= order + 1 or keep.sum() == use.sum():
            break
        use = keep
        fit = Chebyshev.fit(x[use], y[use], order)
    return fit(x)


def correct_baseline(spectrum: Spectrum, order=5, mask=None, window=None, factor=3.0) -> Spectrum:
    """Return a copy of the spectrum with a polynomial baseline subtracted from the real part.

    ``mask`` may be given to choose the baseline points yourself (True = baseline); otherwise
    they are found automatically. The imaginary part is left unchanged.
    """
    y = spectrum.real
    if mask is None:
        mask = find_baseline_points(y, window=window, factor=factor)
    base = fit_baseline(y, mask, order=order)
    info = dict(spectrum.info)
    info["baseline_order"] = order
    return Spectrum(
        data=(y - base) + 1j * spectrum.data.imag,
        ppm=spectrum.ppm,
        sw_hz=spectrum.sw_hz,
        sf_mhz=spectrum.sf_mhz,
        nucleus=spectrum.nucleus,
        info=info,
    )
