"""Main window: open a Bruker 1D data set, process it interactively and view the spectrum."""
from __future__ import annotations

import math
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from nmrproc.export import integrals_to_csv, peaks_to_csv, spectrum_to_csv
from nmrproc.gui.session import COMMON_REFERENCES, ZERO_FILL_CHOICES, Session
from nmrproc.gui.sessionfile import (
    EXTENSION,
    SessionFileError,
    apply_session_data,
    read_session_file,
    resolve_data_path,
    save_session,
)
from nmrproc.gui.widgets import SliderSpin
from nmrproc.integration import integral_curve

MAX_PEAK_LABELS = 80  # label at most this many peaks in view, tallest first
MIN_LABEL_SPACING_PX = 12  # skip a label that would overlap a taller peak's label

WINDOW_CHOICES = (  # (menu text, session name)
    ("None", "none"),
    ("Exponential (line broadening)", "exponential"),
    ("Gaussian (resolution enhancement)", "gaussian"),
    ("Sine bell", "sine"),
    ("Squared sine bell", "qsine"),
)
WINDOW_HELP = {
    "none": "No weighting: maximum resolution but the most noise and truncation ripples.",
    "exponential": "Smooths noise at the cost of wider peaks. The usual choice, especially for "
                   "weak signals such as 13C.",
    "gaussian": "Narrows peaks to separate overlapping multiplets, at the cost of more noise. "
                "Use a negative line broadening (about -1 Hz) and a Gaussian maximum around 0.1.",
    "sine": "Smooth rise and fall that suppresses truncation ripples. Common for 2D data.",
    "qsine": "Squared sine bell: like the sine bell with stronger suppression of ripples.",
}

MODE_NONE, MODE_PIVOT, MODE_REFERENCE, MODE_INTEGRATE = "none", "pivot", "reference", "integrate"
MODE_HINTS = {
    MODE_PIVOT: "Pivot mode: click the spectrum where first-order phase should pivot.  (Esc to cancel)",
    MODE_REFERENCE: "Reference mode: click on a peak to use as the reference.  (Esc to cancel)",
    MODE_INTEGRATE: "Integrate mode: drag across a peak to add a range; drag a range's edge to "
                    "adjust it; click inside a range to select it.  (Esc to finish)",
}
EDGE_TOLERANCE_PX = 7  # how close (in screen pixels) a press must be to a range edge to grab it
CLICK_TOLERANCE_PX = 4  # a press-release closer than this is a click, not a drag
PEAK_CLICK_PX = 10  # how far from a peak (in screen pixels) a reference click may land


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.session = Session()
        self._spectrum = None
        self._peaks = []
        self._line = None
        self._pivot_line = None
        self._peak_artists = []
        self._integral_artists = []
        self._mode = MODE_NONE
        self._drag = None  # state of an integral being drawn or an edge being dragged
        self._rubber = None  # temporary shading shown while drawing a new range
        self._experiment_name = "spectrum"
        self._last_dir = str(Path.home())  # where the 'open data' dialog starts
        self._save_dir = str(Path.home())  # where save and export dialogs start
        self._session_path = None  # the session file in use, if any
        self._saved_snapshot = None  # what the session looked like when last loaded or saved
        self.setWindowTitle("nmrproc")
        self.resize(1250, 760)
        self.setAcceptDrops(True)
        self._build_menu()
        self._build_ui()
        self.statusBar().showMessage("Open a Bruker experiment folder (File > Open) to begin.")

    # ---- construction ------------------------------------------------------------------

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")

        def action(text, handler, shortcut=None):
            item = QtGui.QAction(text, self)
            if shortcut is not None:
                item.setShortcut(shortcut)
            item.triggered.connect(lambda _checked=False: handler())
            return item

        self.open_action = action("&Open experiment folder...", self.open_dialog,
                                  QtGui.QKeySequence.StandardKey.Open)
        self.open_session_action = action("Open &session...", self.open_session_dialog,
                                          QtGui.QKeySequence("Ctrl+Shift+O"))
        self.save_action = action("&Save session", self.save_session,
                                  QtGui.QKeySequence.StandardKey.Save)
        self.save_as_action = action("Save session &as...", self.save_session_as,
                                     QtGui.QKeySequence("Ctrl+Shift+S"))
        self.export_spectrum_action = action("Export processed &spectrum (CSV)...",
                                             self.export_spectrum)
        self.export_peaks_action = action("Export &peak list (CSV)...", self.export_peaks)
        self.export_integrals_action = action("Export &integrals (CSV)...",
                                              self.export_integrals)
        quit_action = action("&Quit", self.close, QtGui.QKeySequence.StandardKey.Quit)
        for item in (self.save_action, self.save_as_action, self.export_spectrum_action,
                     self.export_peaks_action, self.export_integrals_action):
            item.setEnabled(False)
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.open_session_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_spectrum_action)
        file_menu.addAction(self.export_peaks_action)
        file_menu.addAction(self.export_integrals_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.addWidget(self._build_controls())
        layout.addLayout(self._build_plot(), 1)
        self.setCentralWidget(central)

    def _build_plot(self):
        self.figure = Figure(layout="constrained")
        self.ax = self.figure.add_subplot()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self._style_axes()
        self.ax.callbacks.connect("xlim_changed", self._on_xlim_changed)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("resize_event", self._on_resize)
        self.mode_label = QtWidgets.QLabel("")
        self.mode_label.setWordWrap(True)
        self.mode_label.setStyleSheet("background: #fff3c4; border: 1px solid #e0c36a; "
                                      "padding: 4px; font-weight: bold;")
        self.mode_label.hide()
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Escape), self,
                        activated=lambda: self._set_mode(MODE_NONE))
        box = QtWidgets.QVBoxLayout()
        box.addWidget(self.toolbar)
        box.addWidget(self.mode_label)
        box.addWidget(self.canvas, 1)
        return box

    def _build_controls(self):
        tabs = QtWidgets.QTabWidget()
        tabs.setFixedWidth(395)
        tabs.addTab(self._scrollable(self._build_process_tab()), "Process")
        tabs.addTab(self._scrollable(self._build_analyze_tab()), "Analyze")
        self.tabs = tabs
        self._sync_controls()
        self._set_controls_enabled(False)
        return tabs

    @staticmethod
    def _scrollable(page):
        """Wrap a tab's contents in a scroll area so small screens scroll instead of squashing."""
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setWidget(page)
        return area

    def _build_process_tab(self):
        page = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(page)

        data_box = QtWidgets.QGroupBox("Data")
        v = QtWidgets.QVBoxLayout(data_box)
        self.open_button = QtWidgets.QPushButton("Open experiment folder...")
        self.open_button.clicked.connect(self.open_dialog)
        self.info_label = QtWidgets.QLabel("No data loaded")
        self.info_label.setWordWrap(True)
        v.addWidget(self.open_button)
        v.addWidget(self.info_label)
        col.addWidget(data_box)

        proc_box = QtWidgets.QGroupBox("Processing")
        proc_layout = QtWidgets.QVBoxLayout(proc_box)
        window_form = QtWidgets.QFormLayout()
        self.proc_form = QtWidgets.QFormLayout()
        self.window_combo = QtWidgets.QComboBox()
        for text, name in WINDOW_CHOICES:
            self.window_combo.addItem(text, name)
        self.window_combo.currentIndexChanged.connect(self._on_window)
        self.window_help = QtWidgets.QLabel("")
        self.window_help.setWordWrap(True)
        self.window_help.setStyleSheet("color: #555;")
        self.lb_spin = QtWidgets.QDoubleSpinBox()
        self.lb_spin.setDecimals(2)
        self.lb_spin.setSingleStep(0.1)
        self.lb_spin.setSuffix(" Hz")
        self.lb_spin.valueChanged.connect(self._on_lb)
        self.gb_spin = QtWidgets.QDoubleSpinBox()
        self.gb_spin.setRange(0.01, 1.0)
        self.gb_spin.setDecimals(2)
        self.gb_spin.setSingleStep(0.01)
        self.gb_spin.setToolTip("Position of the Gaussian maximum, as a fraction of the "
                                "acquisition time (0.01 to 1).")
        self.gb_spin.valueChanged.connect(self._on_gb)
        self.ssb_spin = QtWidgets.QDoubleSpinBox()
        self.ssb_spin.setRange(0.0, 16.0)
        self.ssb_spin.setDecimals(1)
        self.ssb_spin.setSingleStep(0.5)
        self.ssb_spin.setToolTip("Sine-bell start (TopSpin's SSB): 2 gives a cosine-like decay, "
                                 "larger values start nearer zero, 0 gives a pure sine.")
        self.ssb_spin.valueChanged.connect(self._on_ssb)
        self.zf_combo = QtWidgets.QComboBox()
        for factor in ZERO_FILL_CHOICES:
            self.zf_combo.addItem(f"x{factor}", factor)
        self.zf_combo.currentIndexChanged.connect(self._on_zero_fill)
        window_form.addRow("Window function", self.window_combo)
        self.proc_form.addRow("Line broadening", self.lb_spin)
        self.proc_form.addRow("Gaussian maximum (GB)", self.gb_spin)
        self.proc_form.addRow("Sine-bell offset (SSB)", self.ssb_spin)
        self.proc_form.addRow("Zero filling", self.zf_combo)
        proc_layout.addLayout(window_form)
        proc_layout.addWidget(self.window_help)
        proc_layout.addLayout(self.proc_form)
        col.addWidget(proc_box)

        phase_box = QtWidgets.QGroupBox("Phase")
        v = QtWidgets.QVBoxLayout(phase_box)
        self.p0_slider = SliderSpin("Zero order (at pivot)", -180.0, 180.0)
        self.p1_slider = SliderSpin("First order", -360.0, 360.0)
        self.p0_slider.valueChanged.connect(self._on_p0)
        self.p1_slider.valueChanged.connect(self._on_p1)
        row = QtWidgets.QHBoxLayout()
        self.auto_phase_button = QtWidgets.QPushButton("Auto phase")
        self.reset_phase_button = QtWidgets.QPushButton("Reset")
        self.auto_phase_button.clicked.connect(self._on_auto_phase)
        self.reset_phase_button.clicked.connect(self._on_reset_phase)
        row.addWidget(self.auto_phase_button)
        row.addWidget(self.reset_phase_button)
        self.pivot_check = QtWidgets.QCheckBox("Click spectrum to set pivot")
        self.pivot_check.toggled.connect(lambda on: self._set_mode(MODE_PIVOT if on else MODE_NONE))
        self.phase_label = QtWidgets.QLabel("")
        self.phase_label.setWordWrap(True)
        v.addWidget(self.p0_slider)
        v.addWidget(self.p1_slider)
        v.addLayout(row)
        v.addWidget(self.pivot_check)
        v.addWidget(self.phase_label)
        col.addWidget(phase_box)

        base_box = QtWidgets.QGroupBox("Baseline")
        form = QtWidgets.QFormLayout(base_box)
        self.baseline_check = QtWidgets.QCheckBox("Correct baseline")
        self.baseline_check.toggled.connect(self._on_baseline)
        self.order_spin = QtWidgets.QSpinBox()
        self.order_spin.setRange(1, 12)
        self.order_spin.valueChanged.connect(self._on_baseline_order)
        form.addRow(self.baseline_check)
        form.addRow("Polynomial order", self.order_spin)
        col.addWidget(base_box)

        view_box = QtWidgets.QGroupBox("View")
        v = QtWidgets.QVBoxLayout(view_box)
        self.autofit_check = QtWidgets.QCheckBox("Fit height to visible region")
        self.autofit_check.setChecked(True)
        self.autofit_check.toggled.connect(self._on_autofit)
        self.reset_view_button = QtWidgets.QPushButton("Show full spectrum")
        self.reset_view_button.clicked.connect(self.reset_view)
        v.addWidget(self.autofit_check)
        v.addWidget(self.reset_view_button)
        col.addWidget(view_box)

        col.addStretch(1)
        return page

    def _build_analyze_tab(self):
        page = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(page)

        ref_box = QtWidgets.QGroupBox("Chemical-shift reference")
        v = QtWidgets.QVBoxLayout(ref_box)
        row = QtWidgets.QHBoxLayout()
        self.reference_button = QtWidgets.QPushButton("Set reference: click a peak")
        self.reference_button.setCheckable(True)
        self.reference_button.toggled.connect(
            lambda on: self._set_mode(MODE_REFERENCE if on else MODE_NONE))
        self.clear_reference_button = QtWidgets.QPushButton("Clear")
        self.clear_reference_button.clicked.connect(self._on_clear_reference)
        row.addWidget(self.reference_button, 1)
        row.addWidget(self.clear_reference_button)
        self.reference_label = QtWidgets.QLabel("")
        self.reference_label.setWordWrap(True)
        v.addLayout(row)
        v.addWidget(self.reference_label)
        col.addWidget(ref_box)

        peak_box = QtWidgets.QGroupBox("Peaks")
        form = QtWidgets.QFormLayout(peak_box)
        self.peaks_check = QtWidgets.QCheckBox("Show peak labels")
        self.peaks_check.toggled.connect(self._on_show_peaks)
        self.snr_spin = QtWidgets.QDoubleSpinBox()
        self.snr_spin.setRange(1.0, 1000.0)
        self.snr_spin.setDecimals(1)
        self.snr_spin.valueChanged.connect(self._on_snr)
        form.addRow(self.peaks_check)
        form.addRow("Signal-to-noise threshold", self.snr_spin)
        col.addWidget(peak_box)

        int_box = QtWidgets.QGroupBox("Integrals")
        v = QtWidgets.QVBoxLayout(int_box)
        self.integrate_button = QtWidgets.QPushButton("Draw integrals: drag across a peak")
        self.integrate_button.setCheckable(True)
        self.integrate_button.toggled.connect(
            lambda on: self._set_mode(MODE_INTEGRATE if on else MODE_NONE))
        row = QtWidgets.QHBoxLayout()
        self.suggest_button = QtWidgets.QPushButton("Suggest ranges")
        self.suggest_button.clicked.connect(self._on_suggest_ranges)
        self.clear_ranges_button = QtWidgets.QPushButton("Clear all")
        self.clear_ranges_button.clicked.connect(self._on_clear_ranges)
        row.addWidget(self.suggest_button)
        row.addWidget(self.clear_ranges_button)
        self.integral_table = QtWidgets.QTableWidget(0, 3)
        self.integral_table.setHorizontalHeaderLabels(["From (ppm)", "To (ppm)", "Value"])
        self.integral_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.integral_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.integral_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.integral_table.horizontalHeader().setStretchLastSection(True)
        self.integral_table.verticalHeader().setVisible(False)
        self.integral_table.setMinimumHeight(170)
        self.integral_table.itemSelectionChanged.connect(self._on_integral_selection)
        row2 = QtWidgets.QHBoxLayout()
        self.ref_value_spin = QtWidgets.QDoubleSpinBox()
        self.ref_value_spin.setRange(0.01, 10000.0)
        self.ref_value_spin.setDecimals(2)
        self.ref_value_spin.setValue(1.0)
        self.ref_value_spin.setToolTip("The value the reference integral is set to, e.g. 3.00 "
                                       "for a CH3 group.")
        self.set_ref_integral_button = QtWidgets.QPushButton("Set selected to this value")
        self.set_ref_integral_button.clicked.connect(self._on_set_reference_integral)
        row2.addWidget(self.ref_value_spin)
        row2.addWidget(self.set_ref_integral_button, 1)
        self.delete_range_button = QtWidgets.QPushButton("Delete selected")
        self.delete_range_button.clicked.connect(self._on_delete_range)
        v.addWidget(self.integrate_button)
        v.addLayout(row)
        v.addWidget(self.integral_table)
        v.addLayout(row2)
        v.addWidget(self.delete_range_button)
        col.addWidget(int_box)

        col.addStretch(1)
        return page

    def _style_axes(self):
        self.ax.set_yticks([])
        for side in ("left", "right", "top"):
            self.ax.spines[side].set_visible(False)
        self.ax.set_xlabel("Chemical shift (ppm)")

    # ---- loading -----------------------------------------------------------------------

    def open_dialog(self):
        if not self._confirm_discard():
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose a Bruker experiment folder (contains 'fid' and 'acqus')", self._last_dir
        )
        if path:
            self.load_path(path)

    def load_path(self, path) -> bool:
        """Load an experiment folder; returns False (after telling the user) if that fails."""
        try:
            fid = self.session.load(path)
        except Exception as exc:  # noqa: BLE001 - any read problem is shown to the user
            QtWidgets.QMessageBox.critical(
                self,
                "Could not open data",
                f"{path}\n\n{exc}\n\nPlease choose a 1D Bruker experiment folder: the one that "
                "directly contains the files 'fid' and 'acqus'.",
            )
            return False
        self._last_dir = str(Path(path).parent)
        self._experiment_name = Path(path).parent.name or Path(path).name
        title = fid.title.splitlines()[0] if fid.title else ""
        self.info_label.setText(
            f"{title}\n{fid.nucleus}, {fid.bf_mhz:.2f} MHz, "
            f"{fid.sw_hz / fid.bf_mhz:.1f} ppm wide, {len(fid.data)} complex points".strip()
        )
        self.ax.set_xlabel(f"Chemical shift, {fid.nucleus} (ppm)" if fid.nucleus else
                           "Chemical shift (ppm)")
        self._set_mode(MODE_NONE)
        self._set_controls_enabled(True)
        self._sync_controls()
        self._session_path = None
        self._saved_snapshot = self.session.snapshot()  # a freshly opened data set is not "work"
        self.refresh(reset_view=True)
        self.statusBar().showMessage(f"Loaded {path}")
        return True

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls or not self._confirm_discard():
            return
        dropped = urls[0].toLocalFile()
        if dropped.lower().endswith(EXTENSION):
            self.open_session(dropped)
        else:
            self.load_path(dropped)

    # ---- drawing -----------------------------------------------------------------------

    def refresh(self, reset_view: bool = False):
        """Recompute the spectrum from the current settings and redraw."""
        if not self.session.loaded:
            return
        spec = self.session.spectrum()
        self._spectrum = spec
        self._peaks = self.session.peaks(spec)
        if self._line is None:
            (self._line,) = self.ax.plot(spec.ppm, spec.real, color="black", linewidth=0.8)
        else:
            self._line.set_data(spec.ppm, spec.real)
        if reset_view:
            self.ax.set_xlim(spec.ppm[0], spec.ppm[-1])  # descending: high ppm on the left
        if reset_view or self.autofit_check.isChecked():
            self._fit_height()
        self._update_pivot_marker()
        self._draw_peaks()
        self._refresh_integrals()
        self._update_phase_label()
        self._update_reference_label()
        self._update_title()
        self.canvas.draw_idle()

    def reset_view(self):
        if self._spectrum is not None:
            self.ax.set_xlim(self._spectrum.ppm[0], self._spectrum.ppm[-1])
            self._fit_height()
            self.canvas.draw_idle()

    def _fit_height(self):
        spec = self._spectrum
        if spec is None:
            return
        lo, hi = sorted(self.ax.get_xlim())
        visible = (spec.ppm >= lo) & (spec.ppm <= hi)
        if not visible.any():
            return
        segment = spec.real[visible]
        top, bottom = float(segment.max()), float(segment.min())
        span = (top - bottom) or 1.0
        self.ax.set_ylim(bottom - 0.05 * span, top + 0.25 * span)  # headroom for labels

    def _draw_peaks(self):
        for artist in self._peak_artists:
            artist.remove()
        self._peak_artists = []
        lo, hi = sorted(self.ax.get_xlim())
        in_view = [p for p in self._peaks if lo <= p.ppm <= hi]
        in_view.sort(key=lambda p: -p.height)
        placed = []  # screen x positions of labels already drawn, so labels never overlap
        for p in in_view:
            x_px = self.ax.transData.transform((p.ppm, 0.0))[0]
            if any(abs(x_px - other) < MIN_LABEL_SPACING_PX for other in placed):
                continue
            placed.append(x_px)
            self._peak_artists.append(
                self.ax.annotate(
                    f"{p.ppm:.3f}", xy=(p.ppm, p.height), xytext=(0, 4),
                    textcoords="offset points", rotation=90, ha="center", va="bottom",
                    fontsize=7,
                )
            )
            if len(placed) >= MAX_PEAK_LABELS:
                break

    def _draw_integrals(self):
        """Shade each integration range, draw its running integral and label its value."""
        for artist in self._integral_artists:
            artist.remove()
        self._integral_artists = []
        spec = self._spectrum
        if spec is None or not self.session.regions:
            return
        integrals = self.session.integrals(spec)
        biggest = max(abs(i.area) for i in integrals)
        if biggest == 0:
            return
        bottom, top = self.ax.get_ylim()
        height = top - bottom
        scale = 0.30 * height / biggest  # the largest integral rises 30 % of the plot height
        base = bottom + 0.06 * height
        selected = self._selected_row()
        for index, (region, integral) in enumerate(zip(self.session.regions, integrals)):
            ppm, cumulative = integral_curve(spec, region.start_ppm, region.end_ppm)
            if len(ppm) == 0:
                continue
            chosen = index == selected
            shade = self.ax.axvspan(region.start_ppm, region.end_ppm, color="tab:blue",
                                    alpha=0.22 if chosen else 0.07, linewidth=0)
            (curve,) = self.ax.plot(ppm, base + scale * cumulative, color="tab:red",
                                    linewidth=1.6 if chosen else 1.0)
            label = self.ax.annotate(
                f"{integral.value:.2f}", xy=(float(ppm.mean()), base + scale * cumulative[-1]),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                color="tab:red", fontsize=9,
            )
            self._integral_artists += [shade, curve, label]

    def _update_pivot_marker(self):
        spec = self._spectrum
        x = float(spec.ppm[round(self.session.settings.pivot * (len(spec.ppm) - 1))])
        if self._pivot_line is None:
            self._pivot_line = self.ax.axvline(x, color="tab:red", linestyle=":", linewidth=1)
        else:
            self._pivot_line.set_xdata([x, x])

    def _update_phase_label(self):
        p0, p1 = self.session.topspin_phase()
        self.phase_label.setText(f"TopSpin-style values: PHC0 = {p0:.1f}°, PHC1 = {p1:.1f}°")

    def _update_reference_label(self):
        shift = self.session.settings.reference_shift
        if shift:
            self.reference_label.setText(f"Axis shifted by {shift:+.4f} ppm.")
        else:
            self.reference_label.setText("Not referenced: 0 ppm is the spectrometer's base "
                                         "frequency.")

    # ---- integration table -------------------------------------------------------------

    def _selected_row(self):
        rows = self.integral_table.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def _update_integral_table(self):
        spec = self._spectrum
        integrals = self.session.integrals(spec) if spec is not None else []
        keep = self._selected_row()
        regions = self.session.regions
        has_reference = any(r.reference for r in regions)
        table = self.integral_table
        table.blockSignals(True)
        table.setRowCount(len(integrals))
        for row, (integral, region) in enumerate(zip(integrals, regions)):
            is_reference = region.reference or (row == 0 and not has_reference)
            cells = (f"{integral.start_ppm:.3f}", f"{integral.end_ppm:.3f}",
                     f"{integral.value:.2f}" + ("  (ref)" if is_reference else ""))
            for column, text in enumerate(cells):
                table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        if keep is not None and keep < len(integrals):
            table.selectRow(keep)
        table.blockSignals(False)

    def _refresh_integrals(self):
        self._update_integral_table()
        self._draw_integrals()

    def _on_integral_selection(self):
        self._draw_integrals()
        self.canvas.draw_idle()

    def _on_suggest_ranges(self):
        if self._spectrum is not None:
            self.session.auto_regions(self._spectrum)
            self._refresh_integrals()
            self.canvas.draw_idle()

    def _on_clear_ranges(self):
        self.session.clear_regions()
        self._refresh_integrals()
        self.canvas.draw_idle()

    def _on_delete_range(self):
        row = self._selected_row()
        if row is not None:
            self.session.remove_region(row)
            self._refresh_integrals()
            self.canvas.draw_idle()

    def _on_set_reference_integral(self):
        row = self._selected_row()
        if row is None:
            self.statusBar().showMessage("Select a row in the integral table first.")
            return
        self.session.set_reference_region(row, self.ref_value_spin.value())
        self._refresh_integrals()
        self.canvas.draw_idle()

    # ---- saving, exporting and sessions ------------------------------------------------

    def _is_dirty(self) -> bool:
        """True if the processing, reference or integrals differ from the last save or load."""
        return (self.session.loaded and self._saved_snapshot is not None
                and self.session.snapshot() != self._saved_snapshot)

    def _update_title(self):
        if not self.session.loaded:
            self.setWindowTitle("nmrproc")
            return
        label = self._session_path.name if self._session_path else self._experiment_name
        self.setWindowTitle(f"nmrproc - {label}" + (" *" if self._is_dirty() else ""))

    def _ask_unsaved(self) -> str:
        """Ask what to do about unsaved work: returns 'save', 'discard' or 'cancel'."""
        box = QtWidgets.QMessageBox
        answer = box.question(
            self, "Unsaved work",
            "Your processing settings and integrals have not been saved.\n\n"
            "Save them to a session file first?",
            box.StandardButton.Save | box.StandardButton.Discard | box.StandardButton.Cancel,
            box.StandardButton.Save)
        return {box.StandardButton.Save: "save",
                box.StandardButton.Discard: "discard"}.get(answer, "cancel")

    def _confirm_discard(self) -> bool:
        """True if it is fine to replace or close the current work (asking if it is unsaved)."""
        if not self._is_dirty():
            return True
        choice = self._ask_unsaved()
        if choice == "save":
            return self.save_session()
        return choice == "discard"

    def closeEvent(self, event):
        if self._confirm_discard():
            event.accept()
        else:
            event.ignore()

    def _ask_save_path(self, title, default_name, file_filter):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, title, str(Path(self._save_dir) / default_name), file_filter)
        if path:
            self._save_dir = str(Path(path).parent)
        return path

    def save_session(self, path=None) -> bool:
        """Save to the session file in use (or ask for one). Returns True if it was saved."""
        if not self.session.loaded:
            return False
        path = path or self._session_path
        if path is None:
            return self.save_session_as()
        return self._write_session(Path(path))

    def save_session_as(self, path=None) -> bool:
        if not self.session.loaded:
            return False
        path = path or self._ask_save_path("Save session", f"{self._experiment_name}{EXTENSION}",
                                           f"nmrproc sessions (*{EXTENSION})")
        if not path:
            return False
        path = Path(path)
        if path.suffix.lower() != EXTENSION:
            path = path.with_name(path.name + EXTENSION)
        return self._write_session(path)

    def _write_session(self, path: Path) -> bool:
        left, right = self.ax.get_xlim()
        try:
            save_session(self.session, path, view={"xlim": [float(left), float(right)]})
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Could not save session", f"{path}\n\n{exc}")
            return False
        self._session_path = path
        self._saved_snapshot = self.session.snapshot()
        self._save_dir = str(path.parent)
        self._update_title()
        self.statusBar().showMessage(f"Saved session to {path}")
        return True

    def open_session_dialog(self):
        if not self._confirm_discard():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open session", self._save_dir, f"nmrproc sessions (*{EXTENSION})")
        if path:
            self.open_session(path)

    def _ask_locate_data(self, missing) -> str | None:
        """The session's data folder was not found: let the user point to it."""
        QtWidgets.QMessageBox.information(
            self, "Data folder not found",
            f"The data folder this session used could not be found:\n{missing}\n\n"
            "Please choose the experiment folder (the one containing 'fid' and 'acqus').")
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Locate the experiment folder", self._last_dir)
        return path or None

    def open_session(self, path) -> bool:
        """Reopen a saved session: load its data and restore its settings and integrals.

        The file is checked against a scratch copy first, so a bad file never disturbs the work
        that is currently open.
        """
        path = Path(path)
        try:
            data = read_session_file(path)
        except SessionFileError as exc:
            QtWidgets.QMessageBox.critical(self, "Could not open session", f"{path}\n\n{exc}")
            return False
        data_path = resolve_data_path(data, path) or self._ask_locate_data(data.get("data_path"))
        if data_path is None:
            return False
        try:
            probe = Session()
            probe.load(data_path)
            apply_session_data(probe, data)
        except SessionFileError as exc:
            QtWidgets.QMessageBox.critical(self, "Could not open session", f"{path}\n\n{exc}")
            return False
        except Exception as exc:  # noqa: BLE001 - any read problem is shown to the user
            QtWidgets.QMessageBox.critical(
                self, "Could not open data",
                f"{data_path}\n\n{exc}\n\nThe session's data folder could not be read.")
            return False
        if not self.load_path(data_path):
            return False
        apply_session_data(self.session, data)
        self._sync_controls()
        self.refresh(reset_view=True)
        self._restore_view(data.get("view"))
        self._session_path = path
        self._saved_snapshot = self.session.snapshot()
        self._save_dir = str(path.parent)
        self._update_title()
        self.statusBar().showMessage(f"Opened session {path}")
        return True

    def _restore_view(self, view):
        limits = view.get("xlim") if isinstance(view, dict) else None
        if (isinstance(limits, list) and len(limits) == 2
                and all(isinstance(v, int | float) and math.isfinite(v) for v in limits)
                and limits[0] != limits[1]):
            self.ax.set_xlim(limits[0], limits[1])

    def _export(self, writer, path, title, suffix, message) -> bool:
        if self._spectrum is None:
            return False
        path = path or self._ask_save_path(title, f"{self._experiment_name}_{suffix}.csv",
                                           "CSV files (*.csv)")
        if not path:
            return False
        try:
            writer(path)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Could not save file", f"{path}\n\n{exc}")
            return False
        self.statusBar().showMessage(f"Saved {message} to {path}")
        return True

    def export_spectrum(self, path=None) -> bool:
        """Write the processed spectrum (ppm and intensity) as CSV."""
        return self._export(lambda p: spectrum_to_csv(p, self._spectrum), path,
                            "Export processed spectrum", "spectrum", "the processed spectrum")

    def export_peaks(self, path=None) -> bool:
        if self._spectrum is None:
            return False
        return self._export(lambda p: peaks_to_csv(p, self.session.all_peaks(self._spectrum)),
                            path, "Export peak list", "peaks", "the peak list")

    def export_integrals(self, path=None) -> bool:
        if self._spectrum is None:
            return False
        if not self.session.regions:
            self.statusBar().showMessage("There are no integrals to export yet.")
            return False
        return self._export(
            lambda p: integrals_to_csv(p, self.session.integrals(self._spectrum)),
            path, "Export integrals", "integrals", "the integrals")

    # ---- interaction modes and plot events ---------------------------------------------

    def _set_mode(self, mode, message=None):
        """Only one click/drag mode (pivot, reference, integrate) is active at a time."""
        self._mode = mode
        self._cancel_drag()
        buttons = ((self.pivot_check, MODE_PIVOT), (self.reference_button, MODE_REFERENCE),
                   (self.integrate_button, MODE_INTEGRATE))
        for widget, widget_mode in buttons:
            widget.blockSignals(True)
            widget.setChecked(mode == widget_mode)
            widget.blockSignals(False)
        if mode != MODE_NONE:
            self._release_toolbar()  # a zoom/pan tool left switched on would swallow the clicks
        self._show_banner(message or MODE_HINTS.get(mode, ""))
        self.canvas.setCursor(QtCore.Qt.CursorShape.CrossCursor if mode != MODE_NONE
                              else QtCore.Qt.CursorShape.ArrowCursor)

    def _cancel_drag(self):
        self._drag = None
        if self._rubber is not None:
            self._rubber.remove()
            self._rubber = None

    def _show_banner(self, text):
        """A message above the plot that stays until the next action (unlike the status bar)."""
        self.mode_label.setText(text)
        self.mode_label.setVisible(bool(text))

    def _toolbar_mode(self) -> str:
        mode = getattr(self.toolbar, "mode", "")
        return str(getattr(mode, "value", mode))

    def _toolbar_busy(self) -> bool:
        return bool(self._toolbar_mode())  # the zoom or pan tool is switched on

    def _release_toolbar(self):
        """Switch off the toolbar's zoom/pan tool, if it is on."""
        mode = self._toolbar_mode()
        if "zoom" in mode:
            self.toolbar.zoom()
        elif "pan" in mode:
            self.toolbar.pan()

    def _on_xlim_changed(self, _ax):
        if self._spectrum is None:
            return
        if self.autofit_check.isChecked():
            self._fit_height()
        self._draw_peaks()
        self._draw_integrals()
        self.canvas.draw_idle()

    def _on_resize(self, _event):
        if self._spectrum is not None:
            self._draw_peaks()

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.button != 1 or self._spectrum is None:
            return
        if self._toolbar_busy():
            if self._mode != MODE_NONE:  # the user picked zoom/pan while a mode was on
                self._set_mode(MODE_NONE, "The zoom/pan tool is on, so the click mode was "
                                          "switched off.")
            return
        if self._mode == MODE_INTEGRATE:
            self._start_integral_drag(event)
        elif self._mode != MODE_NONE:
            self._on_click(event)

    def _on_motion(self, event):
        if event.inaxes is not self.ax or event.xdata is None or self._spectrum is None:
            return
        hz = event.xdata * self._spectrum.sf_mhz
        self.statusBar().showMessage(f"{event.xdata:.4f} ppm   ({hz:.1f} Hz)")
        if self._mode == MODE_INTEGRATE:
            if self._drag is not None:
                self._continue_integral_drag(event)
            else:
                grabbed = self._edge_at(event.x) is not None
                self.canvas.setCursor(QtCore.Qt.CursorShape.SizeHorCursor if grabbed
                                      else QtCore.Qt.CursorShape.CrossCursor)

    def _on_release(self, event):
        if self._drag is not None:
            self._finish_integral_drag(event)

    def _on_click(self, event):
        """A click in pivot or reference mode (integrate mode is handled by dragging)."""
        if event.inaxes is not self.ax or event.button != 1 or self._toolbar_busy():
            return
        if self._spectrum is None:
            return
        if self._mode == MODE_PIVOT:
            index = abs(self._spectrum.ppm - event.xdata).argmin()
            self.session.set_pivot(index / (len(self._spectrum.ppm) - 1))
            self._sync_controls()
            self.refresh()
        elif self._mode == MODE_REFERENCE:
            self._reference_at(event.xdata)

    # ---- drawing and adjusting integration ranges with the mouse -----------------------

    def _edge_at(self, x_px):
        """(range index, 'start' or 'end') if screen position ``x_px`` is on a range edge."""
        tolerance = EDGE_TOLERANCE_PX * self.canvas.devicePixelRatioF()
        best = None
        for index, region in enumerate(self.session.regions):
            for edge, ppm in (("start", region.start_ppm), ("end", region.end_ppm)):
                distance = abs(self.ax.transData.transform((ppm, 0.0))[0] - x_px)
                if distance <= tolerance and (best is None or distance < best[0]):
                    best = (distance, index, edge)
        return None if best is None else (best[1], best[2])

    def _start_integral_drag(self, event):
        grabbed = self._edge_at(event.x)
        if grabbed is not None:
            self._drag = {"kind": "edge", "index": grabbed[0], "edge": grabbed[1],
                          "x_px": event.x, "ppm": event.xdata}
            self.integral_table.selectRow(grabbed[0])
        else:
            self._drag = {"kind": "new", "anchor": event.xdata, "ppm": event.xdata,
                          "x_px": event.x}

    def _continue_integral_drag(self, event):
        drag = self._drag
        drag["ppm"] = event.xdata
        if drag["kind"] == "new":
            if self._rubber is not None:
                self._rubber.remove()
            self._rubber = self.ax.axvspan(drag["anchor"], event.xdata, color="tab:orange",
                                           alpha=0.25, linewidth=0)
        else:
            drag["index"] = self.session.move_region_edge(
                self._spectrum, drag["index"], drag["edge"], event.xdata)
            self._refresh_integrals()
            self.integral_table.selectRow(drag["index"])
        self.canvas.draw_idle()

    def _finish_integral_drag(self, event):
        drag, self._drag = self._drag, None
        if self._rubber is not None:
            self._rubber.remove()
            self._rubber = None
        ppm = event.xdata if event.xdata is not None else drag["ppm"]
        moved = abs((event.x if event.x is not None else drag["x_px"]) - drag["x_px"])
        if drag["kind"] == "new":
            if moved < CLICK_TOLERANCE_PX * self.canvas.devicePixelRatioF():
                self._select_range_at(drag["anchor"])  # a click, not a drag
            else:
                region = self.session.add_region(self._spectrum, drag["anchor"], ppm)
                if region is None:
                    self._show_banner("That range is too narrow to integrate.")
                else:
                    self._show_banner(MODE_HINTS[MODE_INTEGRATE])
                    self._refresh_integrals()
                    self.integral_table.selectRow(self.session.regions.index(region))
        self.canvas.draw_idle()

    def _select_range_at(self, ppm):
        index = self.session.region_at(ppm)
        if index is None:
            self.integral_table.clearSelection()
        else:
            self.integral_table.selectRow(index)

    # ---- referencing -------------------------------------------------------------------

    def _reference_at(self, ppm):
        x_low, x_high = sorted(self.ax.get_xlim())
        ppm_per_px = (x_high - x_low) / (self.ax.bbox.width or 1.0)
        spacing = abs(float(self._spectrum.ppm[1] - self._spectrum.ppm[0]))
        window = max(PEAK_CLICK_PX * self.canvas.devicePixelRatioF() * ppm_per_px, 3 * spacing)
        peak = self.session.nearest_peak(self._spectrum, ppm, window)
        if peak is None:
            self._show_banner("No peak found at that spot. Click right on top of a peak, or "
                              "zoom in first (then switch the zoom tool off again).")
            return
        target = self._ask_reference(peak.ppm)
        self._set_mode(MODE_NONE)
        if target is None:
            return
        self.session.reference_to(peak.ppm, target)
        self._show_banner("")
        self.refresh()
        self.statusBar().showMessage(f"The peak at {peak.ppm:.4f} ppm is now at {target:.4f} ppm.")

    def _ask_reference(self, observed_ppm):
        """Ask which shift the clicked peak should have; returns ppm, or None if cancelled."""
        nucleus = self.session.fid.nucleus
        options = [(label, ppm) for label, nuc, ppm in COMMON_REFERENCES if nuc == nucleus]
        options.sort(key=lambda item: abs(item[1] - observed_ppm))
        items = [f"{ppm:g}  {label}" for label, ppm in options] or [f"{observed_ppm:.4f}"]
        text, accepted = QtWidgets.QInputDialog.getItem(
            self, "Set chemical-shift reference",
            f"The peak you clicked is at {observed_ppm:.4f} ppm.\nChoose a standard or type the "
            "shift (ppm) it should have:", items, 0, True)
        if not accepted:
            return None
        try:
            return float(text.split()[0])
        except (ValueError, IndexError):
            QtWidgets.QMessageBox.warning(self, "Not a number",
                                          "Please enter the reference shift as a number in ppm.")
            return None

    def _on_clear_reference(self):
        self.session.clear_reference()
        self.refresh()

    # ---- control handlers --------------------------------------------------------------

    def _on_window(self, index):
        self.session.set_window(self.window_combo.itemData(index))
        self._sync_controls()
        self.refresh()

    def _on_lb(self, value):
        self.session.settings.lb_hz = value
        self.refresh()

    def _on_gb(self, value):
        self.session.settings.gb = value
        self.refresh()

    def _on_ssb(self, value):
        self.session.settings.ssb = value
        self.refresh()

    def _on_zero_fill(self, index):
        self.session.settings.zero_fill = self.zf_combo.itemData(index)
        self.refresh()

    def _on_p0(self, value):
        self.session.settings.p0 = value
        self.refresh()

    def _on_p1(self, value):
        self.session.settings.p1 = value
        self.refresh()

    def _on_auto_phase(self):
        self.session.auto_phase()
        self._sync_controls()
        self.refresh()

    def _on_reset_phase(self):
        self.session.reset_phase()
        self._sync_controls()
        self.refresh()

    def _on_baseline(self, checked):
        self.session.settings.baseline = checked
        self.refresh()

    def _on_baseline_order(self, value):
        self.session.settings.baseline_order = value
        if self.session.settings.baseline:
            self.refresh()

    def _on_show_peaks(self, checked):
        self.session.settings.show_peaks = checked
        self.refresh()

    def _on_snr(self, value):
        self.session.settings.peak_snr = value
        if self.session.settings.show_peaks:
            self.refresh()

    def _on_autofit(self, checked):
        if checked and self._spectrum is not None:
            self._fit_height()
            self.canvas.draw_idle()

    # ---- helpers -----------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool):
        for widget in (self.window_combo, self.lb_spin, self.gb_spin, self.ssb_spin,
                       self.zf_combo, self.p0_slider, self.p1_slider, self.auto_phase_button,
                       self.reset_phase_button, self.pivot_check, self.baseline_check,
                       self.order_spin, self.autofit_check, self.reset_view_button,
                       self.reference_button, self.clear_reference_button, self.peaks_check,
                       self.snr_spin, self.integrate_button, self.suggest_button,
                       self.clear_ranges_button, self.integral_table, self.ref_value_spin,
                       self.set_ref_integral_button, self.delete_range_button):
            widget.setEnabled(enabled)
        for item in (self.save_action, self.save_as_action, self.export_spectrum_action,
                     self.export_peaks_action, self.export_integrals_action):
            item.setEnabled(enabled)

    def _sync_controls(self):
        """Show the session's settings in the controls without triggering their handlers."""
        s = self.session.settings
        widgets = (self.window_combo, self.lb_spin, self.gb_spin, self.ssb_spin, self.zf_combo,
                   self.order_spin, self.snr_spin, self.baseline_check, self.peaks_check)
        for w in widgets:
            w.blockSignals(True)
        self.window_combo.setCurrentIndex(self.window_combo.findData(s.window))
        # Gaussian needs a negative line broadening; the other windows a positive one.
        if s.window == "gaussian":
            self.lb_spin.setRange(-50.0, 0.0)
        else:
            self.lb_spin.setRange(0.0, 50.0)
        self.lb_spin.setValue(s.lb_hz)
        self.gb_spin.setValue(s.gb)
        self.ssb_spin.setValue(s.ssb)
        self.zf_combo.setCurrentIndex(ZERO_FILL_CHOICES.index(s.zero_fill))
        self.order_spin.setValue(s.baseline_order)
        self.snr_spin.setValue(s.peak_snr)
        self.baseline_check.setChecked(s.baseline)
        self.peaks_check.setChecked(s.show_peaks)
        for w in widgets:
            w.blockSignals(False)
        for widget, shown in ((self.lb_spin, s.window in ("exponential", "gaussian")),
                              (self.gb_spin, s.window == "gaussian"),
                              (self.ssb_spin, s.window in ("sine", "qsine"))):
            self.proc_form.setRowVisible(widget, shown)
        self.window_help.setText(WINDOW_HELP[s.window])
        self.p0_slider.set_value(s.p0)
        self.p1_slider.set_value(s.p1)
        if self._spectrum is not None:
            self._update_phase_label()
