"""Tests of the GUI's processing session (no Qt needed)."""
from pathlib import Path

import numpy as np
import pytest

from nmrproc.gui.session import ZERO_FILL_CHOICES, Session, wrap_angle
from nmrproc.io import load_fid_1d, load_processed_1d
from nmrproc.processing import process_1d

DATA = Path(__file__).resolve().parent.parent / "testdata"
PATH = DATA / "triphenylphosphine-1H" / "10"

pytestmark = pytest.mark.skipif(not PATH.exists(), reason="testdata/ folder not present")


def loaded_session():
    session = Session()
    session.load(PATH)
    return session


def test_wrap_angle():
    assert wrap_angle(190.0) == pytest.approx(-170.0)
    assert wrap_angle(-190.0) == pytest.approx(170.0)
    assert wrap_angle(30.0) == pytest.approx(30.0)


def test_load_sets_defaults_and_pivot_on_tallest_peak():
    session = loaded_session()
    assert session.loaded
    assert 0.0 < session.settings.pivot < 1.0
    spec = session.spectrum()
    assert len(spec.ppm) == session.si == 2 * 32768
    assert int(np.argmax(np.abs(session._unphased().data))) == pytest.approx(
        session.settings.pivot * session.si, abs=1
    )


def test_load_missing_folder_raises():
    with pytest.raises((OSError, ValueError)):
        Session().load(DATA / "does-not-exist")


def test_zero_filling_changes_number_of_points():
    session = loaded_session()
    base = session.si
    session.settings.zero_fill = 4
    assert session.si == 2 * base
    assert len(session.spectrum().ppm) == session.si
    assert set(ZERO_FILL_CHOICES) >= {1, 2, 4, 8}


def test_session_phase_matches_topspin_convention():
    session = loaded_session()
    s = session.settings
    s.p0, s.p1, s.pivot = 40.0, 25.0, 0.3
    p0_ts, p1_ts = session.topspin_phase()
    assert (p0_ts, p1_ts) == (pytest.approx(40.0 - 25.0 * 0.3), 25.0)
    expected = process_1d(load_fid_1d(PATH), lb_hz=s.lb_hz, si=session.si, p0=p0_ts, p1=p1_ts)
    assert np.allclose(session.spectrum().data, expected.data)


def test_set_pivot_keeps_the_displayed_spectrum_unchanged():
    session = loaded_session()
    session.settings.p0, session.settings.p1 = 33.0, 80.0
    before = session.spectrum().data.copy()
    session.set_pivot(0.7)
    assert session.settings.pivot == pytest.approx(0.7)
    assert np.allclose(session.spectrum().data, before)
    session.set_pivot(5.0)  # out-of-range values are clamped
    assert session.settings.pivot == 1.0


def test_reset_phase():
    session = loaded_session()
    session.settings.p0, session.settings.p1 = 33.0, 80.0
    session.reset_phase()
    assert (session.settings.p0, session.settings.p1) == (0.0, 0.0)


def test_auto_phase_and_baseline_match_topspin():
    session = loaded_session()
    session.auto_phase()
    session.settings.baseline = True
    ref = load_processed_1d(PATH)
    ours = session.spectrum().real
    assert len(ours) == len(ref.real)
    assert np.corrcoef(ours, ref.real)[0, 1] > 0.998


def test_peaks_can_be_switched_off_and_respond_to_threshold():
    session = loaded_session()
    session.auto_phase()
    spec = session.spectrum()
    many = session.peaks(spec)
    session.settings.peak_snr = 200.0
    fewer = session.peaks(spec)
    assert len(fewer) < len(many)
    session.settings.show_peaks = False
    assert session.peaks(spec) == []


# ---- window functions, referencing and integration ranges ----------------------------------


def test_window_choice_changes_the_spectrum_and_adjusts_line_broadening():
    session = loaded_session()
    exp_spec = session.spectrum().real
    session.set_window("gaussian")
    assert session.settings.lb_hz < 0  # Gaussian needs negative LB; set automatically
    gauss_spec = session.spectrum().real
    assert not np.allclose(exp_spec, gauss_spec)
    session.set_window("exponential")
    assert session.settings.lb_hz > 0
    session.set_window("qsine")
    assert session.settings.lb_hz > 0  # untouched for windows that do not use LB
    session.set_window("none")
    assert len(session.spectrum().ppm) == session.si


def test_auto_phase_follows_the_chosen_window():
    session = loaded_session()
    session.set_window("qsine")
    session.auto_phase()
    assert (session.settings.p0, session.settings.p1) != (0.0, 0.0)


def test_referencing_moves_the_axis_and_can_be_cleared():
    session = loaded_session()
    session.auto_phase()
    spec = session.spectrum()
    peak = session.nearest_peak(spec, 7.28)
    assert peak.ppm == pytest.approx(7.284, abs=0.005)
    session.reference_to(peak.ppm, 7.26)
    moved = session.spectrum()
    assert moved.ppm[int(np.argmax(moved.real))] == pytest.approx(7.26, abs=0.002)
    assert session.settings.reference_shift == pytest.approx(7.26 - peak.ppm)
    assert np.allclose(moved.real, spec.real)  # only the axis moved
    session.clear_reference()
    assert session.settings.reference_shift == 0.0
    assert np.allclose(session.spectrum().ppm, spec.ppm)


def test_nearest_peak_returns_none_in_empty_regions():
    session = loaded_session()
    session.auto_phase()
    assert session.nearest_peak(session.spectrum(), 12.0, window_ppm=0.1) is None


def test_regions_are_sorted_clipped_and_must_be_wide_enough():
    session = loaded_session()
    spec = session.spectrum()
    session.add_region(spec, 1.7, 1.4)
    session.add_region(spec, 7.1, 7.9)  # given low-to-high on purpose
    assert [round(r.start_ppm, 2) for r in session.regions] == [7.9, 1.7]
    assert session.add_region(spec, 5.0, 5.0 + 1e-6) is None  # too narrow
    clipped = session.add_region(spec, 100.0, 90.0)
    assert clipped is None or clipped.start_ppm <= float(spec.ppm.max())
    session.remove_region(0)
    assert len(session.regions) == 1 and session.regions[0].start_ppm == pytest.approx(1.7)
    session.clear_regions()
    assert session.regions == [] and session.integrals(spec) == []


def test_integrals_are_calibrated_to_the_reference_region():
    session = loaded_session()
    session.auto_phase()
    session.settings.baseline = True
    spec = session.spectrum()
    session.add_region(spec, 7.79, 7.25)
    session.add_region(spec, 1.80, 1.40)
    values = [i.value for i in session.integrals(spec)]
    assert values[0] == pytest.approx(1.0)  # no reference marked: first range = 1.00
    session.set_reference_region(1, 3.0)
    values = [i.value for i in session.integrals(spec)]
    assert values[1] == pytest.approx(3.0)
    assert values[0] > 10 * values[1]  # the aromatic region dwarfs the small water signal
    assert session.regions[1].reference and not session.regions[0].reference


def test_regions_follow_the_peaks_when_referencing():
    session = loaded_session()
    session.auto_phase()
    spec = session.spectrum()
    session.add_region(spec, 7.5, 7.1)
    before = [i.area for i in session.integrals(spec)]
    session.reference_to(7.284, 7.26)
    after = [i.area for i in session.integrals(session.spectrum())]
    assert session.regions[0].start_ppm == pytest.approx(7.5 + (7.26 - 7.284))
    assert after == pytest.approx(before, rel=1e-6)  # same points enclosed, same area


def test_load_resets_regions_and_reference():
    session = loaded_session()
    session.add_region(session.spectrum(), 7.5, 7.1)
    session.reference_to(7.284, 7.26)
    session.load(PATH)
    assert session.regions == [] and session.settings.reference_shift == 0.0
    assert session.reference_value == 1.0
