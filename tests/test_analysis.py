"""Tests of phasing, baseline correction, peak picking and integration on synthetic spectra."""
import numpy as np
import pytest

from nmrproc.baseline import baseline_noise, correct_baseline, find_baseline_points
from nmrproc.data import Spectrum
from nmrproc.integration import (
    find_integral_regions,
    integral_trace,
    integrate,
    normalize_integrals,
)
from nmrproc.peaks import pick_peaks
from nmrproc.phasing import auto_phase
from nmrproc.processing import estimate_noise, phase_correct, phase_spectrum, ppm_axis

N, SW, SF, LEFT = 16384, 4000.0, 400.0, 10.0  # 10 ppm wide, 400 MHz


def ppm_to_index(ppm):
    return (LEFT - ppm) / (SW / SF / N)


def make_spectrum(peaks, noise=0.0, seed=0, half_width_hz=1.5):
    """Complex Lorentzian lines. peaks = [(ppm, amplitude), ...]; absorption in the real part."""
    ppm = ppm_axis(N, SW, SF, LEFT)
    hz = ppm * SF
    data = np.zeros(N, dtype=complex)
    for p, amp in peaks:
        u = (hz - p * SF) / half_width_hz
        data += amp * (1 + 1j * u) / (1 + u**2)
    if noise:
        rng = np.random.default_rng(seed)
        data = data + noise * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    return Spectrum(data=data, ppm=ppm, sw_hz=SW, sf_mhz=SF, nucleus="1H")


PEAKS = [(8.2, 1.0), (6.1, 0.6), (3.4, 0.9), (1.2, 0.3)]


def test_estimate_noise_gaussian():
    rng = np.random.default_rng(1)
    assert estimate_noise(rng.standard_normal(50000)) == pytest.approx(1.0, rel=0.05)


def test_phase_spectrum_tracks_totals_and_undoes_phase():
    spec = make_spectrum(PEAKS)
    once = phase_spectrum(spec, 20.0, 30.0)
    twice = phase_spectrum(once, 10.0, 5.0, pivot=0.5)
    assert twice.info["p0"] == pytest.approx(20.0 + 10.0 - 5.0 * 0.5)
    assert twice.info["p1"] == pytest.approx(35.0)
    back = phase_spectrum(twice, -twice.info["p0"], -twice.info["p1"])
    assert np.allclose(back.data, spec.data)


def wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


@pytest.mark.parametrize("p0,p1,tol", [(70.0, 0.0, 6.0), (-120.0, 40.0, 15.0),
                                       (160.0, -60.0, 15.0), (30.0, 90.0, 15.0)])
def test_auto_phase_recovers_phase(p0, p1, tol):
    """The phase error at each peak (what matters visually) stays within a few degrees."""
    truth = make_spectrum(PEAKS, noise=0.002)
    distorted = Spectrum(
        data=phase_correct(truth.data, -p0, -p1),  # apply the opposite of the correction
        ppm=truth.ppm, sw_hz=SW, sf_mhz=SF,
    )
    a0, a1 = auto_phase(distorted)
    for ppm, _ in PEAKS:
        x = ppm_to_index(ppm) / N
        assert abs(wrap((a0 + a1 * x) - (p0 + p1 * x))) < tol


def test_auto_phase_zero_order_only():
    truth = make_spectrum(PEAKS, noise=0.002)
    distorted = Spectrum(data=phase_correct(truth.data, -50.0, 0.0), ppm=truth.ppm,
                         sw_hz=SW, sf_mhz=SF)
    a0, a1 = auto_phase(distorted, fit_p1=False)
    assert a1 == 0.0
    assert abs(((a0 - 50.0 + 180) % 360) - 180) < 2.0


def baseline_test_spectrum():
    clean = make_spectrum(PEAKS, noise=0.002)
    x = np.linspace(-1, 1, N)
    wobble = 0.03 + 0.02 * x - 0.05 * x**2 + 0.02 * x**3
    spec = Spectrum(data=clean.data.real + wobble + 0j, ppm=clean.ppm, sw_hz=SW, sf_mhz=SF)
    return clean, spec, wobble


def test_find_baseline_points_excludes_peaks():
    _, spec, _ = baseline_test_spectrum()
    mask = find_baseline_points(spec.real)
    for ppm, _ in PEAKS:
        i = round(ppm_to_index(ppm))
        assert not mask[i]
    assert mask.mean() > 0.8


def test_correct_baseline_removes_wobble_and_keeps_peaks():
    clean, spec, wobble = baseline_test_spectrum()
    fixed = correct_baseline(spec, order=5)
    far = ppm_to_index(9.5) <= np.arange(N)  # a signal-free stretch near the right edge
    assert abs(np.mean(fixed.real[far])) < 0.005
    # peak heights unchanged to within 2 %
    for ppm, amp in PEAKS:
        i = round(ppm_to_index(ppm))
        assert fixed.real[i - 3:i + 4].max() == pytest.approx(clean.real[i - 3:i + 4].max(),
                                                             rel=0.02)
    # overall the baseline error is far smaller than the wobble we added
    assert np.std(fixed.real - clean.real) < 0.25 * np.std(wobble)


def test_correct_baseline_accepts_manual_mask():
    _, spec, _ = baseline_test_spectrum()
    mask = np.zeros(N, dtype=bool)
    mask[:200] = True
    mask[-200:] = True
    mask[N // 2 - 100:N // 2 + 100] = True
    mask[N // 4 - 100:N // 4 + 100] = True  # contains no peak here
    out = correct_baseline(spec, order=3, mask=mask)
    assert out.info["baseline_order"] == 3


def test_baseline_noise_matches_added_noise():
    spec = make_spectrum(PEAKS, noise=0.01)
    assert baseline_noise(spec.real) == pytest.approx(0.01, rel=0.15)


def test_pick_peaks_positions_and_threshold():
    peaks_in = [(8.2503, 1.0), (6.1017, 0.6), (3.4, 0.9), (1.2, 0.002)]  # last one is too small
    spec = make_spectrum(peaks_in, noise=0.001)
    found = pick_peaks(spec, snr=8)
    assert len(found) == 3
    assert [round(p.ppm, 2) for p in found] == [8.25, 6.10, 3.40]  # high to low ppm
    point_ppm = SW / SF / N
    for p, (ppm, amp) in zip(found, peaks_in[:3]):
        assert abs(p.ppm - ppm) < 0.2 * point_ppm  # interpolation beats the point spacing
        assert p.height == pytest.approx(amp, rel=0.01)
        assert p.hz == pytest.approx(p.ppm * SF)


def test_pick_peaks_min_rel_height_and_range():
    spec = make_spectrum(PEAKS, noise=0.001)
    assert len(pick_peaks(spec, min_rel_height=0.5)) == 3
    assert [round(p.ppm, 1) for p in pick_peaks(spec, ppm_range=(3.0, 7.0))] == [6.1, 3.4]


def test_integrate_ratio_and_normalize():
    spec = make_spectrum([(7.0, 2.0), (3.0, 1.0)], noise=0.0005)
    ints = integrate(spec, [(7.2, 6.8), (3.2, 2.8)])
    ints = normalize_integrals(ints, which=1, value=3.0)
    assert ints[1].value == pytest.approx(3.0)
    assert ints[0].value == pytest.approx(6.0, rel=0.02)
    assert ints[0].start_ppm > ints[0].end_ppm  # stored high to low
    # ranges may be given in either order
    flipped = integrate(spec, [(6.8, 7.2)])
    assert flipped[0].area == pytest.approx(ints[0].area)


def test_normalize_rejects_zero_reference():
    spec = make_spectrum([(7.0, 1.0)])
    ints = integrate(spec, [(2.0, 1.0)])  # no signal there
    ints[0].area = 0.0
    with pytest.raises(ValueError):
        normalize_integrals(ints)


def test_find_integral_regions_merges_multiplet():
    # two lines 6 Hz apart (a doublet) and one isolated line
    spec = make_spectrum([(7.0, 1.0), (7.015, 1.0), (3.0, 1.0)], noise=0.001)
    regions = find_integral_regions(spec)
    assert len(regions) == 2
    (hi1, lo1), (hi2, lo2) = regions
    assert hi1 > 7.015
    assert lo1 < 7.0
    assert hi2 > 3.0 > lo2


def test_integral_trace_is_flat_outside_ranges():
    spec = make_spectrum([(7.0, 1.0), (3.0, 1.0)])
    trace = integral_trace(spec, [(7.5, 6.5)])
    inside = np.where((spec.ppm <= 7.5) & (spec.ppm >= 6.5))[0]
    assert np.all(trace[:inside[0]] == 0)
    assert np.all(trace[inside[-1]:] == trace[inside[-1]])
    assert trace[inside[-1]] > 0
