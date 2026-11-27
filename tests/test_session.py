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
