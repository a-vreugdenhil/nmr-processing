"""Window functions (apodization), the integral curve helper and CSV export."""
import csv

import numpy as np
import pytest

from nmrproc.data import FID
from nmrproc.export import integrals_to_csv, peaks_to_csv, spectrum_to_csv
from nmrproc.integration import integral_curve, integrate, normalize_integrals
from nmrproc.peaks import Peak
from nmrproc.processing import (
    WINDOWS,
    apodize,
    exponential_apodization,
    gaussian_apodization,
    process_1d,
    sine_apodization,
)

SW, N = 4000.0, 8192
ONES = np.ones(N, dtype=complex)
AQ = N / SW


def window_of(func, *args, **kwargs):
    return func(ONES, *args, **kwargs).real


def test_none_returns_the_signal_unchanged():
    assert apodize(ONES, "none", SW) is ONES


def test_apodize_matches_the_individual_functions():
    assert np.allclose(apodize(ONES, "exponential", SW, lb_hz=2.0),
                       exponential_apodization(ONES, 2.0, SW))
    assert np.allclose(apodize(ONES, "gaussian", SW, lb_hz=-1.5, gb=0.2),
                       gaussian_apodization(ONES, -1.5, 0.2, SW))
    assert np.allclose(apodize(ONES, "sine", SW, ssb=3), sine_apodization(ONES, 3, 1))
    assert np.allclose(apodize(ONES, "qsine", SW, ssb=3), sine_apodization(ONES, 3, 2))


def test_unknown_window_is_rejected():
    assert set(WINDOWS) == {"none", "exponential", "gaussian", "sine", "qsine"}
    with pytest.raises(ValueError):
        apodize(ONES, "hamming", SW)


def test_gaussian_starts_at_one_and_peaks_at_gb_times_aq():
    w = window_of(gaussian_apodization, -2.0, 0.25, SW)
    assert w[0] == pytest.approx(1.0)
    peak_time = np.argmax(w) / SW
    assert peak_time == pytest.approx(0.25 * AQ, rel=0.01)
    assert w.max() > 1.0  # the rising exponential gives a bump above 1 before it falls


def test_gaussian_validates_its_parameters():
    with pytest.raises(ValueError):
        gaussian_apodization(ONES, 1.0, 0.1, SW)  # LB must be negative
    with pytest.raises(ValueError):
        gaussian_apodization(ONES, -1.0, 0.0, SW)
    with pytest.raises(ValueError):
        gaussian_apodization(ONES, -1.0, 1.5, SW)


def test_sine_bell_shapes():
    cos_like = window_of(sine_apodization, 2.0)
    assert cos_like[0] == pytest.approx(1.0)
    assert cos_like[-1] < 1e-3
    assert np.all(np.diff(cos_like) <= 1e-12)  # ssb = 2 falls monotonically
    pure_sine = window_of(sine_apodization, 0.0)
    assert pure_sine[0] == pytest.approx(0.0)
    assert pure_sine[N // 2] == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(window_of(sine_apodization, 1.0), pure_sine)  # TopSpin: 0 and 1 are alike
    later = window_of(sine_apodization, 4.0)
    assert 0 < later[0] < cos_like[0]  # larger ssb starts nearer zero


def test_squared_sine_is_the_square_of_sine():
    assert np.allclose(window_of(sine_apodization, 3.0, 2), window_of(sine_apodization, 3.0) ** 2)


def lorentzian_fid(decay_hz=3.0, freq_hz=0.0):
    t = np.arange(N) / SW
    data = np.exp(2j * np.pi * freq_hz * t) * np.exp(-np.pi * decay_hz * t)
    return FID(data=data, sw_hz=SW, bf_mhz=400.0, sfo_mhz=400.0, o1_hz=0.0, nucleus="1H")


def fwhm_hz(spectrum):
    y = spectrum.real
    above = np.where(y >= y.max() / 2)[0]
    return (above[-1] - above[0] + 1) * spectrum.sw_hz / len(y)


def test_resolution_enhancement_narrows_the_line():
    fid = lorentzian_fid(decay_hz=3.0)
    plain = process_1d(fid, lb_hz=0.0, si=4 * N, window="exponential")
    enhanced = process_1d(fid, lb_hz=-3.0, gb=0.3, si=4 * N, window="gaussian")
    assert fwhm_hz(enhanced) < 0.6 * fwhm_hz(plain)


def test_sine_bell_removes_truncation_wiggles():
    t = np.arange(N) / SW
    fid = FID(data=np.exp(2j * np.pi * 300.0 * t) * np.exp(-np.pi * 0.2 * t), sw_hz=SW,
              bf_mhz=400.0, sfo_mhz=400.0, o1_hz=0.0)  # barely decays: strongly truncated
    far = slice(0, N // 8)  # a stretch of spectrum well away from the line
    plain = process_1d(fid, window="none", si=2 * N)
    bell = process_1d(fid, window="qsine", ssb=2.0, si=2 * N)
    assert np.std(bell.real[far]) / bell.real.max() < 0.2 * np.std(plain.real[far]) / plain.real.max()


def test_process_1d_window_choices_are_recorded_and_applied():
    fid = lorentzian_fid()
    spec = process_1d(fid, window="qsine", ssb=3.0, si=2 * N)
    assert spec.info["window"] == "qsine" and spec.info["ssb"] == 3.0
    assert not np.allclose(spec.data, process_1d(fid, window="none", si=2 * N).data)


def test_default_process_1d_is_still_exponential():
    fid = lorentzian_fid()
    assert np.allclose(process_1d(fid, lb_hz=1.0, si=2 * N).data,
                       process_1d(fid, lb_hz=1.0, si=2 * N, window="exponential").data)


def two_peak_spectrum():
    from test_analysis import make_spectrum  # synthetic Lorentzian spectrum helper

    return make_spectrum([(7.0, 2.0), (3.0, 1.0)])


def test_integral_curve_rises_to_the_integral_area():
    spec = two_peak_spectrum()
    ppm, cumulative = integral_curve(spec, 7.3, 6.7)
    assert np.all(np.diff(ppm) < 0)  # high to low ppm, like the spectrum
    assert np.all(np.diff(cumulative) > -1e-12)  # a positive peak only adds
    assert cumulative[-1] == pytest.approx(integrate(spec, [(7.3, 6.7)])[0].area)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def test_csv_exports(tmp_path):
    spec = two_peak_spectrum()
    integrals = normalize_integrals(integrate(spec, [(7.3, 6.7), (3.3, 2.7)]), which=1, value=3.0)
    peaks_file, int_file, spec_file = (tmp_path / n for n in ("p.csv", "i.csv", "s.csv"))
    peaks_to_csv(peaks_file, [Peak(index=10, ppm=7.0, hz=2800.0, height=1.5)])
    integrals_to_csv(int_file, integrals)
    spectrum_to_csv(spec_file, spec)
    rows = read_csv(peaks_file)
    assert rows[0] == ["ppm", "hz", "height"] and float(rows[1][0]) == pytest.approx(7.0)
    rows = read_csv(int_file)
    assert rows[0] == ["start_ppm", "end_ppm", "area", "value"]
    assert [float(r[3]) for r in rows[1:]] == pytest.approx([6.0, 3.0], rel=0.02)
    rows = read_csv(spec_file)
    assert len(rows) == len(spec.ppm) + 1 and rows[0] == ["ppm", "real", "imaginary"]
