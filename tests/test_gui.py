"""Smoke and behaviour tests of the Qt window, run offscreen (skipped if Qt cannot be loaded)."""
import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

from nmrproc.gui.main_window import MainWindow

PATH = Path(__file__).resolve().parent.parent / "testdata" / "triphenylphosphine-1H" / "10"

pytestmark = pytest.mark.skipif(not PATH.exists(), reason="testdata/ folder not present")


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(qapp):
    win = MainWindow()
    win.show()
    assert win.load_path(PATH)
    qapp.processEvents()
    yield win
    win.close()


def ydata(win):
    return np.asarray(win._line.get_ydata())


def test_controls_disabled_until_data_is_loaded(qapp):
    win = MainWindow()
    assert not win.auto_phase_button.isEnabled()
    assert not win.lb_spin.isEnabled()
    win.close()


def test_opening_data_shows_a_spectrum_with_high_ppm_on_the_left(window):
    assert len(window._line.get_xdata()) == window.session.si
    left, right = window.ax.get_xlim()
    assert left > right
    assert "1H" in window.info_label.text()
    assert window.auto_phase_button.isEnabled()
    assert "nmrproc" in window.windowTitle()


def test_line_broadening_changes_the_spectrum(window):
    before = ydata(window).copy()
    window.lb_spin.setValue(3.0)
    assert window.session.settings.lb_hz == pytest.approx(3.0)
    assert not np.allclose(before, ydata(window))


def test_zero_fill_combo_changes_number_of_points(window):
    n = len(window._line.get_xdata())
    window.zf_combo.setCurrentIndex(window.zf_combo.findData(4))
    assert len(window._line.get_xdata()) == 2 * n


def test_phase_sliders_update_spectrum_and_label(window):
    before = ydata(window).copy()
    window.p0_slider.spin.setValue(45.0)
    assert window.session.settings.p0 == pytest.approx(45.0)
    assert not np.allclose(before, ydata(window))
    window.p1_slider.slider.setValue(1000)  # slider units are tenths of a degree
    assert window.session.settings.p1 == pytest.approx(100.0)
    assert "PHC1" in window.phase_label.text()


def test_auto_phase_button_sets_the_sliders_and_improves_the_phase(window):
    unphased_min = ydata(window).min()
    window.auto_phase_button.click()
    s = window.session.settings
    assert (s.p0, s.p1) != (0.0, 0.0)
    assert window.p0_slider.value() == pytest.approx(s.p0, abs=0.05)
    assert window.p1_slider.value() == pytest.approx(s.p1, abs=0.05)
    assert ydata(window).min() > unphased_min  # the large negative dispersive lobe is gone
    window.reset_phase_button.click()
    assert window.session.settings.p0 == 0.0 and window.p0_slider.value() == 0.0


def test_baseline_checkbox(window):
    window.auto_phase_button.click()
    before = ydata(window).copy()
    window.baseline_check.setChecked(True)
    assert window.session.settings.baseline
    assert not np.allclose(before, ydata(window))


def test_peak_labels_follow_zoom_and_checkbox(window):
    window.auto_phase_button.click()
    window.ax.set_xlim(7.5, 7.15)
    assert 1 <= len(window._peak_artists) <= 80
    window.peaks_check.setChecked(False)
    assert window._peak_artists == []
    window.peaks_check.setChecked(True)
    assert len(window._peak_artists) >= 1


def test_labels_do_not_overlap(window):
    window.auto_phase_button.click()
    window.ax.set_xlim(7.40, 7.30)
    xs = sorted(window.ax.transData.transform(a.xy)[0] for a in window._peak_artists)
    assert all(b - a >= 12 for a, b in itertools.pairwise(xs))


def test_height_follows_the_visible_region(window):
    window.auto_phase_button.click()
    full_top = window.ax.get_ylim()[1]
    window.ax.set_xlim(2.0, 1.2)  # only the small water peak is in view
    assert window.ax.get_ylim()[1] < 0.1 * full_top
    window.reset_view_button.click()
    assert window.ax.get_ylim()[1] == pytest.approx(full_top)


def test_clicking_sets_the_pivot_without_changing_the_spectrum(window):
    window.auto_phase_button.click()
    before = ydata(window).copy()
    window.pivot_check.setChecked(True)
    window._on_click(SimpleNamespace(inaxes=window.ax, button=1, xdata=1.57))
    assert window.session.settings.pivot == pytest.approx((16.41 - 1.57) / 20.5, abs=0.01)
    assert np.allclose(before, ydata(window))


def test_bad_folder_reports_an_error_instead_of_crashing(window, monkeypatch, tmp_path):
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args: shown.append(args))
    assert window.load_path(tmp_path) is False
    assert len(shown) == 1
    assert window.session.loaded  # the previous data set is still there
