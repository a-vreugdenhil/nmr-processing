"""Smoke and behaviour tests of the Qt window, run offscreen (skipped if Qt cannot be loaded)."""
import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)
from PySide6 import QtCore, QtGui
from PySide6.QtTest import QTest

from nmrproc.gui.main_window import MainWindow

PATH = Path(__file__).resolve().parent.parent / "testdata" / "triphenylphosphine-1H" / "10"

pytestmark = pytest.mark.skipif(not PATH.exists(), reason="testdata/ folder not present")


@pytest.fixture(autouse=True)
def never_block_on_the_unsaved_work_dialog(monkeypatch):
    """Tests close windows with unsaved changes; answer the prompt instead of waiting for a click."""
    monkeypatch.setattr(MainWindow, "_ask_unsaved", lambda self: "discard")


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


LEFT = QtCore.Qt.MouseButton.LeftButton
NO_MODIFIER = QtCore.Qt.KeyboardModifier.NoModifier


def canvas_point(win, ppm, height=0.5):
    """Qt position on the plot canvas of a chemical shift at a fraction of the axes height."""
    bottom, top = win.ax.get_ylim()
    x, y = win.ax.transData.transform((ppm, bottom + height * (top - bottom)))
    ratio = win.canvas.devicePixelRatioF()
    return QtCore.QPoint(round(x / ratio), round((win.canvas.figure.bbox.height - y) / ratio))


def mouse_click(win, ppm, height=0.5):
    QTest.mouseClick(win.canvas, LEFT, NO_MODIFIER, canvas_point(win, ppm, height))
    QtWidgets.QApplication.processEvents()


def mouse_drag(win, ppm_from, ppm_to, height=0.5):
    """A real press, a few moves and a release on the plot, like a user dragging."""
    start, end = canvas_point(win, ppm_from, height), canvas_point(win, ppm_to, height)
    QTest.mousePress(win.canvas, LEFT, NO_MODIFIER, start)
    for fraction in (0.25, 0.5, 0.75):
        step = QtCore.QPoint(round(start.x() + (end.x() - start.x()) * fraction), start.y())
        QTest.mouseMove(win.canvas, step)
        QtWidgets.QApplication.processEvents()
    QTest.mouseMove(win.canvas, end)
    QTest.mouseRelease(win.canvas, LEFT, NO_MODIFIER, end)
    QtWidgets.QApplication.processEvents()


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
    peak = tallest_peak_ppm(window)
    mouse_click(window, peak + 0.004)  # a click about a pixel off the peak top, as a mouse is
    assert asked and asked[0] == pytest.approx(7.284, abs=0.005)
    assert tallest_peak_ppm(window) == pytest.approx(7.26, abs=0.002)
    assert "shifted" in window.reference_label.text()
    assert window._mode == "none" and not window.reference_button.isChecked()
    window.clear_reference_button.click()
    assert tallest_peak_ppm(window) == pytest.approx(7.284, abs=0.005)
    assert "Not referenced" in window.reference_label.text()


def test_reference_click_picks_the_closest_peak_not_the_tallest(window, monkeypatch):
    window.auto_phase_button.click()
    window.ax.set_xlim(7.45, 7.15)
    QtWidgets.QApplication.processEvents()
    asked = []
    monkeypatch.setattr(window, "_ask_reference", lambda observed: asked.append(observed) or None)
    window.reference_button.click()
    mouse_click(window, 7.334, 0.3)  # on a small multiplet line, 0.05 ppm from the big singlet
    assert asked and asked[0] == pytest.approx(7.334, abs=0.01)


def test_reference_click_away_from_any_peak_explains_itself(window, monkeypatch):
    window.auto_phase_button.click()
    monkeypatch.setattr(window, "_ask_reference", lambda observed: pytest.fail("should not ask"))
    window.reference_button.click()
    mouse_click(window, 12.0)
    assert window.session.settings.reference_shift == 0.0
    assert window._mode == "reference"  # still waiting for a proper click
    assert "No peak found" in window.mode_label.text()


def test_cancelling_the_reference_dialog_changes_nothing(window, monkeypatch):
    window.auto_phase_button.click()
    monkeypatch.setattr(window, "_ask_reference", lambda observed: None)
    window.reference_button.click()
    mouse_click(window, tallest_peak_ppm(window))
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
    assert window._mode == "pivot"
    window.integrate_button.click()
    assert window._mode == "integrate"
    assert not window.pivot_check.isChecked()
    window.reference_button.click()
    assert window._mode == "reference"
    assert not window.integrate_button.isChecked()
    window.reference_button.click()  # clicking it again switches the mode off
    assert window._mode == "none"


def test_a_banner_shows_the_active_mode_and_escape_cancels_it(window):
    assert window.mode_label.isHidden()
    window.integrate_button.click()
    assert not window.mode_label.isHidden() and "Integrate mode" in window.mode_label.text()
    window.reference_button.click()
    assert "Reference mode" in window.mode_label.text()
    QTest.keyClick(window, QtCore.Qt.Key.Key_Escape)
    assert window._mode == "none" and window.mode_label.isHidden()
    assert not window.reference_button.isChecked()


def test_dragging_in_integrate_mode_adds_an_integral(window):
    prepare_integration(window)
    mouse_drag(window, 7.62, 7.25)
    mouse_drag(window, 1.75, 1.40)
    table = window.integral_table
    assert table.rowCount() == 2
    assert float(table.item(0, 0).text()) == pytest.approx(7.62, abs=0.05)
    assert float(table.item(0, 1).text()) == pytest.approx(7.25, abs=0.05)
    assert "(ref)" in table.item(0, 2).text()
    assert float(table.item(0, 2).text().split()[0]) == pytest.approx(1.0)
    assert 0 < float(table.item(1, 2).text()) < 0.2  # the small water signal
    assert len(window._integral_artists) == 6  # shading, curve and label for each range
    mouse_drag(window, 5.0, 5.0 + 0.001)  # far too narrow to be a range
    assert table.rowCount() == 2


def test_dragging_works_when_zoomed_in(window):
    prepare_integration(window)
    window.ax.set_xlim(7.5, 7.15)
    QtWidgets.QApplication.processEvents()
    mouse_drag(window, 7.40, 7.30)
    assert window.integral_table.rowCount() == 1
    assert float(window.integral_table.item(0, 0).text()) == pytest.approx(7.40, abs=0.01)


def test_dragging_a_range_edge_adjusts_the_range(window):
    prepare_integration(window)
    mouse_drag(window, 7.62, 7.25)
    before = window.integral_table.item(0, 2).text()
    mouse_drag(window, 7.25, 7.10)  # starts on the existing range's low-ppm edge
    assert window.integral_table.rowCount() == 1  # adjusted, not a second range
    assert float(window.integral_table.item(0, 1).text()) == pytest.approx(7.10, abs=0.05)
    assert float(window.integral_table.item(0, 0).text()) == pytest.approx(7.62, abs=0.05)
    mouse_drag(window, 7.62, 7.70)  # and the high-ppm edge
    assert float(window.integral_table.item(0, 0).text()) == pytest.approx(7.70, abs=0.05)
    assert window.session.regions[0].start_ppm > window.session.regions[0].end_ppm
    assert before  # the table was populated before the adjustment


def test_clicking_inside_a_range_selects_its_row(window):
    prepare_integration(window)
    mouse_drag(window, 7.62, 7.25)
    mouse_drag(window, 1.75, 1.40)
    window.integral_table.clearSelection()
    mouse_click(window, 1.55)
    assert window._selected_row() == 1
    mouse_click(window, 7.45)
    assert window._selected_row() == 0
    mouse_click(window, 4.5)  # empty spectrum: clears the selection
    assert window._selected_row() is None


def test_zoom_tool_left_on_is_switched_off_when_a_mode_starts(window):
    window.toolbar.zoom()  # the user zoomed in earlier and left the magnifier on
    assert window._toolbar_busy()
    window.integrate_button.click()
    assert not window._toolbar_busy()  # the tool no longer swallows the clicks
    mouse_drag(window, 7.62, 7.25)
    assert window.integral_table.rowCount() == 1


def test_turning_on_the_zoom_tool_during_a_mode_cancels_the_mode_visibly(window):
    window.integrate_button.click()
    window.toolbar.pan()
    mouse_click(window, 7.3)  # a press while the pan tool is active
    assert window._mode == "none" and not window.integrate_button.isChecked()
    assert "zoom/pan tool is on" in window.mode_label.text()
    assert window.integral_table.rowCount() == 0
    window.toolbar.pan()  # tool off again


def test_integral_calibration_deletion_and_clearing(window):
    prepare_integration(window)
    mouse_drag(window, 7.62, 7.25)
    mouse_drag(window, 1.75, 1.40)
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
    mouse_drag(window, 7.62, 7.25)
    mouse_drag(window, 1.75, 1.40)
    before = window.integral_table.item(1, 2).text()
    window.p0_slider.spin.setValue(window.p0_slider.value() + 40.0)  # worse phase, new values
    assert window.integral_table.item(1, 2).text() != before
    window.p0_slider.spin.setValue(window.p0_slider.value() - 40.0)
    assert window.integral_table.item(1, 2).text() == before
    monkeypatch.setattr(window, "_ask_reference", lambda observed: 7.26)
    window.reference_button.click()
    mouse_click(window, tallest_peak_ppm(window))
    assert window.integral_table.item(1, 2).text() == before  # ranges moved with the peaks
    assert float(window.integral_table.item(0, 0).text()) == pytest.approx(7.62 - 0.0241, abs=0.05)


def test_selecting_a_row_highlights_its_range(window):
    prepare_integration(window)
    mouse_drag(window, 7.62, 7.25)
    mouse_drag(window, 1.75, 1.40)
    window.integral_table.selectRow(0)
    shaded = [a for a in window._integral_artists if hasattr(a, "get_alpha")
              and a.get_alpha() is not None]
    assert max(a.get_alpha() for a in shaded) > 0.2 > min(a.get_alpha() for a in shaded)


def test_exporting_peaks_and_integrals(window, tmp_path):
    import csv

    prepare_integration(window)
    peaks_file, integrals_file = tmp_path / "peaks.csv", tmp_path / "integrals.csv"
    assert window.export_integrals(str(integrals_file)) is False  # nothing to export yet
    mouse_drag(window, 7.62, 7.25)
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
    mouse_drag(window, 7.62, 7.25)
    assert window.load_path(PATH)
    assert window.integral_table.rowCount() == 0
    assert window._mode == "none" and not window.integrate_button.isChecked()


# ---- saving, sessions and exporting the spectrum -------------------------------------------


def make_some_work(win):
    win.auto_phase_button.click()
    win.baseline_check.setChecked(True)
    win.integrate_button.click()
    mouse_drag(win, 7.62, 7.25)
    mouse_drag(win, 1.75, 1.40)
    win.integral_table.selectRow(1)
    win.ref_value_spin.setValue(2.0)
    win.set_ref_integral_button.click()
    win.ax.set_xlim(7.8, 1.2)
    QtWidgets.QApplication.processEvents()
    win.integrate_button.click()  # leave the mode


def record_messages(monkeypatch, which="critical"):
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, which, lambda *args: shown.append(args))
    return shown


def test_title_marks_unsaved_work_and_a_save_clears_it(window, tmp_path):
    assert window.windowTitle().startswith("nmrproc - ") and not window.windowTitle().endswith("*")
    window.p0_slider.spin.setValue(40.0)
    assert window.windowTitle().endswith(" *")
    assert window.save_session_as(tmp_path / "lab1.nmrproc")
    assert window.windowTitle() == "nmrproc - lab1.nmrproc"
    window.p0_slider.spin.setValue(41.0)
    assert window.windowTitle() == "nmrproc - lab1.nmrproc *"
    assert window.save_session()  # saves to the same file without asking
    assert not window.windowTitle().endswith("*")


def test_zooming_and_looking_around_is_not_unsaved_work(window):
    window.ax.set_xlim(7.5, 7.1)
    window.reset_view()
    window.peaks_check.setChecked(True)  # unchanged value
    assert not window._is_dirty()


def test_save_as_adds_the_extension_and_remembers_the_folder(window, tmp_path, monkeypatch):
    chosen = tmp_path / "my lab"  # no extension typed
    monkeypatch.setattr(window, "_ask_save_path", lambda *args: str(chosen))
    window.p0_slider.spin.setValue(20.0)
    assert window.save_session()  # no session file yet, so it asks where to save
    assert (tmp_path / "my lab.nmrproc").exists()
    assert window._session_path.name == "my lab.nmrproc"


def test_cancelling_the_save_dialog_saves_nothing(window, monkeypatch):
    monkeypatch.setattr(window, "_ask_save_path", lambda *args: "")
    window.p0_slider.spin.setValue(20.0)
    assert window.save_session() is False and window._is_dirty()


def test_a_saved_session_reopens_with_everything_restored(window, qapp, tmp_path):
    make_some_work(window)
    saved = tmp_path / "work.nmrproc"
    assert window.save_session_as(saved)
    expected = window.session.snapshot()
    other = MainWindow()
    other.show()
    try:
        assert other.open_session(saved)
        assert other.session.snapshot() == expected
        assert other.windowTitle() == "nmrproc - work.nmrproc"
        assert not other._is_dirty()
        assert other.integral_table.rowCount() == 2
        assert other.integral_table.item(1, 2).text() == "2.00  (ref)"
        assert other.baseline_check.isChecked()
        assert other.p0_slider.value() == pytest.approx(window.session.settings.p0, abs=0.05)
        assert other.ax.get_xlim() == pytest.approx((7.8, 1.2))  # the zoom comes back too
        assert np.allclose(ydata(other), ydata(window))
        assert other.window_combo.currentData() == window.window_combo.currentData()
    finally:
        other.close()


def test_closing_a_clean_window_does_not_ask(window, monkeypatch):
    asked = []
    monkeypatch.setattr(MainWindow, "_ask_unsaved", lambda self: asked.append(1) or "cancel")
    event = QtGui.QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() and asked == []


@pytest.mark.parametrize("answer,closes", [("cancel", False), ("discard", True), ("save", True)])
def test_closing_with_unsaved_work_asks_what_to_do(window, monkeypatch, tmp_path, answer, closes):
    monkeypatch.setattr(MainWindow, "_ask_unsaved", lambda self: answer)
    monkeypatch.setattr(window, "_ask_save_path", lambda *args: str(tmp_path / "closing.nmrproc"))
    window.p0_slider.spin.setValue(33.0)
    event = QtGui.QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted() is closes
    assert (tmp_path / "closing.nmrproc").exists() is (answer == "save")


def test_closing_stays_open_if_the_save_is_cancelled(window, monkeypatch):
    monkeypatch.setattr(MainWindow, "_ask_unsaved", lambda self: "save")
    monkeypatch.setattr(window, "_ask_save_path", lambda *args: "")  # user backs out of the dialog
    window.p0_slider.spin.setValue(33.0)
    event = QtGui.QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()


def test_opening_other_data_asks_first_and_cancel_keeps_the_work(window, monkeypatch):
    monkeypatch.setattr(MainWindow, "_ask_unsaved", lambda self: "cancel")
    chooser = QtWidgets.QFileDialog
    monkeypatch.setattr(chooser, "getExistingDirectory",
                        lambda *a, **k: pytest.fail("must not open the file dialog"))
    window.p0_slider.spin.setValue(33.0)
    window.open_dialog()
    assert window.session.settings.p0 == pytest.approx(33.0)


def test_a_broken_session_file_is_reported_and_current_work_is_kept(window, monkeypatch, tmp_path):
    shown = record_messages(monkeypatch)
    window.p0_slider.spin.setValue(33.0)
    bad = tmp_path / "bad.nmrproc"
    bad.write_text("this is not a session", encoding="utf-8")
    assert window.open_session(bad) is False
    assert len(shown) == 1 and "not a valid nmrproc session" in str(shown[0])
    assert window.session.settings.p0 == pytest.approx(33.0)  # untouched


def test_a_session_with_invalid_settings_is_rejected_before_anything_changes(
        window, monkeypatch, tmp_path):
    import json

    shown = record_messages(monkeypatch)
    window.save_session_as(tmp_path / "ok.nmrproc")
    path = tmp_path / "ok.nmrproc"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["settings"]["window"] = "hamming"
    path.write_text(json.dumps(payload), encoding="utf-8")
    window.p0_slider.spin.setValue(33.0)
    assert window.open_session(path) is False
    assert len(shown) == 1 and "unknown window function" in str(shown[0])
    assert window.session.settings.p0 == pytest.approx(33.0)


def test_a_session_whose_data_has_moved_asks_where_it_went(window, monkeypatch, tmp_path):
    import json

    saved = tmp_path / "lost.nmrproc"
    window.p0_slider.spin.setValue(25.0)
    window.save_session_as(saved)
    payload = json.loads(saved.read_text(encoding="utf-8"))
    payload["data_path"] = str(tmp_path / "vanished")
    payload["data_path_relative"] = "vanished"
    saved.write_text(json.dumps(payload), encoding="utf-8")
    other = MainWindow()
    try:
        monkeypatch.setattr(other, "_ask_locate_data", lambda missing: None)
        assert other.open_session(saved) is False  # user declines to locate it
        assert not other.session.loaded
        monkeypatch.setattr(other, "_ask_locate_data", lambda missing: str(PATH))
        assert other.open_session(saved) is True
        assert other.session.settings.p0 == pytest.approx(25.0)
    finally:
        other.close()


def test_dropping_a_session_file_opens_it(window, tmp_path):
    window.p0_slider.spin.setValue(25.0)
    saved = tmp_path / "dropped.nmrproc"
    window.save_session_as(saved)
    other = MainWindow()
    try:
        mime = QtCore.QMimeData()
        mime.setUrls([QtCore.QUrl.fromLocalFile(str(saved))])
        event = QtGui.QDropEvent(QtCore.QPointF(5, 5), QtCore.Qt.DropAction.CopyAction, mime,
                                 LEFT, NO_MODIFIER)
        other.dropEvent(event)
        assert other.session.loaded and other.session.settings.p0 == pytest.approx(25.0)
    finally:
        other.close()


def test_exporting_the_processed_spectrum(window, tmp_path):
    import csv

    window.auto_phase_button.click()
    window.baseline_check.setChecked(True)
    out = tmp_path / "spectrum.csv"
    assert window.export_spectrum(str(out))
    with open(out, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["ppm", "intensity"] and len(rows) == len(window._spectrum.ppm) + 1
    top = max(range(1, len(rows)), key=lambda i: float(rows[i][1]))
    assert float(rows[top][0]) == pytest.approx(tallest_peak_ppm(window), abs=1e-4)


def test_export_to_an_unwritable_place_reports_the_problem(window, monkeypatch, tmp_path):
    shown = record_messages(monkeypatch)
    assert window.export_spectrum(str(tmp_path / "no-such-folder" / "out.csv")) is False
    assert len(shown) == 1


def test_save_and_export_menu_items_need_data(qapp):
    win = MainWindow()
    try:
        for item in (win.save_action, win.save_as_action, win.export_spectrum_action,
                     win.export_peaks_action, win.export_integrals_action):
            assert not item.isEnabled()
        assert win.open_session_action.isEnabled() and win.open_action.isEnabled()
        assert win.save_session() is False and win.export_spectrum() is False
        win.load_path(PATH)
        assert win.save_action.isEnabled() and win.export_spectrum_action.isEnabled()
    finally:
        win.close()


def test_loading_new_data_forgets_the_old_session_file(window, tmp_path):
    window.save_session_as(tmp_path / "first.nmrproc")
    assert window.load_path(PATH)
    assert window._session_path is None and not window._is_dirty()
