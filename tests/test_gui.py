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


# ---- window functions, referencing and integration -----------------------------------------


def visible_rows(win):
    """Which window-function parameter rows are currently shown."""
    return {
        "lb": not win.lb_spin.isHidden(),
        "gb": not win.gb_spin.isHidden(),
        "ssb": not win.ssb_spin.isHidden(),
    }


def choose_window(win, name):
    win.window_combo.setCurrentIndex(win.window_combo.findData(name))


def test_window_function_choice_shows_the_relevant_parameters(window):
    assert visible_rows(window) == {"lb": True, "gb": False, "ssb": False}  # exponential
    choose_window(window, "gaussian")
    assert visible_rows(window) == {"lb": True, "gb": True, "ssb": False}
    assert window.lb_spin.value() < 0 and window.lb_spin.maximum() <= 0
    choose_window(window, "sine")
    assert visible_rows(window) == {"lb": False, "gb": False, "ssb": True}
    choose_window(window, "none")
    assert visible_rows(window) == {"lb": False, "gb": False, "ssb": False}
    choose_window(window, "exponential")
    assert window.lb_spin.value() > 0 and window.lb_spin.minimum() >= 0
    assert window.window_help.text()


def test_window_parameters_change_the_spectrum(window):
    choose_window(window, "gaussian")
    before = ydata(window).copy()
    window.gb_spin.setValue(0.4)
    assert window.session.settings.gb == pytest.approx(0.4)
    assert not np.allclose(before, ydata(window))
    choose_window(window, "qsine")
    before = ydata(window).copy()
    window.ssb_spin.setValue(4.0)
    assert window.session.settings.ssb == pytest.approx(4.0)
    assert not np.allclose(before, ydata(window))


def tallest_peak_ppm(win):
    return float(win._spectrum.ppm[int(np.argmax(win._spectrum.real))])


def test_reference_click_moves_the_axis(window, monkeypatch):
    window.auto_phase_button.click()
    asked = []
    monkeypatch.setattr(window, "_ask_reference", lambda observed: asked.append(observed) or 7.26)
    window.reference_button.click()
    assert window._mode == "reference"
    window._on_click(SimpleNamespace(inaxes=window.ax, button=1, xdata=7.28))
    assert asked and asked[0] == pytest.approx(7.284, abs=0.005)
    assert tallest_peak_ppm(window) == pytest.approx(7.26, abs=0.002)
    assert "shifted" in window.reference_label.text()
    assert window._mode == "none" and not window.reference_button.isChecked()
    window.clear_reference_button.click()
    assert tallest_peak_ppm(window) == pytest.approx(7.284, abs=0.005)
    assert "Not referenced" in window.reference_label.text()


def test_reference_click_away_from_any_peak_does_nothing(window, monkeypatch):
    window.auto_phase_button.click()
    monkeypatch.setattr(window, "_ask_reference", lambda observed: pytest.fail("should not ask"))
    window.reference_button.click()
    window._on_click(SimpleNamespace(inaxes=window.ax, button=1, xdata=12.0))
    assert window.session.settings.reference_shift == 0.0
    assert window._mode == "reference"  # still waiting for a proper click


def test_cancelling_the_reference_dialog_changes_nothing(window, monkeypatch):
    window.auto_phase_button.click()
    monkeypatch.setattr(window, "_ask_reference", lambda observed: None)
    window.reference_button.click()
    window._on_click(SimpleNamespace(inaxes=window.ax, button=1, xdata=7.28))
    assert window.session.settings.reference_shift == 0.0 and window._mode == "none"


def test_reference_dialog_offers_standards_for_the_nucleus_and_accepts_typed_values(
        window, monkeypatch):
    seen = {}

    def fake_get_item(parent, title, label, items, current, editable):
        seen["items"] = items
        return "7.30", True

    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", fake_get_item)
    assert window._ask_reference(7.284) == pytest.approx(7.30)
    assert seen["items"][0].startswith("7.26")  # CDCl3 is the closest 1H standard to 7.284
    assert all(item.split()[0].replace(".", "", 1).isdigit() for item in seen["items"])
    assert not any("H3PO4" in item for item in seen["items"])  # a 31P standard is not offered
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", lambda *a: ("abc", True))
    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a: warned.append(a))
    assert window._ask_reference(7.284) is None and len(warned) == 1


def prepare_integration(win):
    win.auto_phase_button.click()
    win.baseline_check.setChecked(True)
    win.integrate_button.click()


def test_only_one_interaction_mode_is_active(window):
    window.pivot_check.setChecked(True)
    assert window._mode == "pivot" and not window._span.active
    window.integrate_button.click()
    assert window._mode == "integrate" and window._span.active
    assert not window.pivot_check.isChecked()
    window.reference_button.click()
    assert window._mode == "reference" and not window._span.active
    assert not window.integrate_button.isChecked()
    window.reference_button.click()  # clicking it again switches the mode off
    assert window._mode == "none"


def test_dragging_in_integrate_mode_adds_an_integral(window):
    prepare_integration(window)
    window._on_span(7.62, 7.25)
    window._on_span(1.75, 1.40)
    table = window.integral_table
    assert table.rowCount() == 2
    assert table.item(0, 0).text() == "7.620" and "(ref)" in table.item(0, 2).text()
    assert float(table.item(0, 2).text().split()[0]) == pytest.approx(1.0)
    assert 0 < float(table.item(1, 2).text()) < 0.2  # the small water signal
    assert len(window._integral_artists) == 6  # shading, curve and label for each range
    window._on_span(5.0, 5.0 + 1e-6)  # too narrow: ignored
    assert table.rowCount() == 2


def test_integral_calibration_deletion_and_clearing(window):
    prepare_integration(window)
    window._on_span(7.62, 7.25)
    window._on_span(1.75, 1.40)
    window.integral_table.selectRow(1)
    window.ref_value_spin.setValue(2.0)
    window.set_ref_integral_button.click()
    assert window.integral_table.item(1, 2).text() == "2.00  (ref)"
    assert float(window.integral_table.item(0, 2).text()) > 10
    window.delete_range_button.click()  # deletes the selected (second) range
    assert window.integral_table.rowCount() == 1
    window.integral_table.selectRow(0)
    window.delete_range_button.click()
    assert window.integral_table.rowCount() == 0 and window._integral_artists == []
    window.set_ref_integral_button.click()  # nothing selected: just a hint, no crash
    window.suggest_button.click()
    assert window.integral_table.rowCount() >= 2
    window.clear_ranges_button.click()
    assert window.integral_table.rowCount() == 0


def test_integrals_follow_phase_changes_and_referencing(window, monkeypatch):
    prepare_integration(window)
    window._on_span(7.62, 7.25)
    window._on_span(1.75, 1.40)
    before = window.integral_table.item(1, 2).text()
    window.p0_slider.spin.setValue(window.p0_slider.value() + 40.0)  # worse phase, new values
    assert window.integral_table.item(1, 2).text() != before
    window.p0_slider.spin.setValue(window.p0_slider.value() - 40.0)
    assert window.integral_table.item(1, 2).text() == before
    monkeypatch.setattr(window, "_ask_reference", lambda observed: 7.26)
    window.reference_button.click()
    window._on_click(SimpleNamespace(inaxes=window.ax, button=1, xdata=7.28))
    assert window.integral_table.item(1, 2).text() == before  # ranges moved with the peaks
    assert float(window.integral_table.item(0, 0).text()) == pytest.approx(7.62 - 0.0241, abs=0.002)


def test_selecting_a_row_highlights_its_range(window):
    prepare_integration(window)
    window._on_span(7.62, 7.25)
    window._on_span(1.75, 1.40)
    window.integral_table.selectRow(0)
    shaded = [a for a in window._integral_artists if hasattr(a, "get_alpha")
              and a.get_alpha() is not None]
    assert max(a.get_alpha() for a in shaded) > 0.2 > min(a.get_alpha() for a in shaded)


def test_exporting_peaks_and_integrals(window, tmp_path):
    import csv

    prepare_integration(window)
    peaks_file, integrals_file = tmp_path / "peaks.csv", tmp_path / "integrals.csv"
    assert window.export_integrals(str(integrals_file)) is False  # nothing to export yet
    window._on_span(7.62, 7.25)
    assert window.export_peaks(str(peaks_file)) and window.export_integrals(str(integrals_file))
    with open(peaks_file, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["ppm", "hz", "height"] and len(rows) > 5
    with open(integrals_file, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["start_ppm", "end_ppm", "area", "value"] and float(rows[1][3]) == 1.0


def test_export_menu_items_need_data(qapp):
    win = MainWindow()
    assert not win.export_peaks_action.isEnabled()
    assert win.export_peaks() is False
    win.close()


def test_loading_a_new_data_set_clears_integrals_and_modes(window):
    prepare_integration(window)
    window._on_span(7.62, 7.25)
    assert window.load_path(PATH)
    assert window.integral_table.rowCount() == 0
    assert window._mode == "none" and not window.integrate_button.isChecked()
