"""Main window: open a Bruker 1D data set, process it interactively and view the spectrum."""
from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtGui, QtWidgets

from nmrproc.gui.session import ZERO_FILL_CHOICES, Session
from nmrproc.gui.widgets import SliderSpin

MAX_PEAK_LABELS = 80  # label at most this many peaks in view, tallest first
MIN_LABEL_SPACING_PX = 12  # skip a label that would overlap a taller peak's label


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.session = Session()
        self._spectrum = None
        self._peaks = []
        self._line = None
        self._pivot_line = None
        self._peak_artists = []
        self._last_dir = str(Path.home())
        self.setWindowTitle("nmrproc")
        self.resize(1200, 700)
        self.setAcceptDrops(True)
        self._build_menu()
        self._build_ui()
        self.statusBar().showMessage("Open a Bruker experiment folder (File > Open) to begin.")

    # ---- construction ------------------------------------------------------------------

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")
        open_action = QtGui.QAction("&Open experiment folder...", self)
        open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_dialog)
        quit_action = QtGui.QAction("&Quit", self)
        quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(open_action)
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
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("resize_event", self._on_resize)
        box = QtWidgets.QVBoxLayout()
        box.addWidget(self.toolbar)
        box.addWidget(self.canvas, 1)
        return box

    def _build_controls(self):
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(320)
        col = QtWidgets.QVBoxLayout(panel)

        # Data
        data_box = QtWidgets.QGroupBox("Data")
        v = QtWidgets.QVBoxLayout(data_box)
        self.open_button = QtWidgets.QPushButton("Open experiment folder...")
        self.open_button.clicked.connect(self.open_dialog)
        self.info_label = QtWidgets.QLabel("No data loaded")
        self.info_label.setWordWrap(True)
        v.addWidget(self.open_button)
        v.addWidget(self.info_label)
        col.addWidget(data_box)

        # Processing
        proc_box = QtWidgets.QGroupBox("Processing")
        form = QtWidgets.QFormLayout(proc_box)
        self.lb_spin = QtWidgets.QDoubleSpinBox()
        self.lb_spin.setRange(0.0, 50.0)
        self.lb_spin.setSingleStep(0.1)
        self.lb_spin.setDecimals(2)
        self.lb_spin.setSuffix(" Hz")
        self.lb_spin.valueChanged.connect(self._on_lb)
        self.zf_combo = QtWidgets.QComboBox()
        for factor in ZERO_FILL_CHOICES:
            self.zf_combo.addItem(f"x{factor}", factor)
        self.zf_combo.currentIndexChanged.connect(self._on_zero_fill)
        form.addRow("Line broadening", self.lb_spin)
        form.addRow("Zero filling", self.zf_combo)
        col.addWidget(proc_box)

        # Phase
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
        self.phase_label = QtWidgets.QLabel("")
        self.phase_label.setWordWrap(True)
        v.addWidget(self.p0_slider)
        v.addWidget(self.p1_slider)
        v.addLayout(row)
        v.addWidget(self.pivot_check)
        v.addWidget(self.phase_label)
        col.addWidget(phase_box)

        # Baseline
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

        # Peaks
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

        # View
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
        self._sync_controls()
        self._set_controls_enabled(False)
        return panel

    def _style_axes(self):
        self.ax.set_yticks([])
        for side in ("left", "right", "top"):
            self.ax.spines[side].set_visible(False)
        self.ax.set_xlabel("Chemical shift (ppm)")

    # ---- loading -----------------------------------------------------------------------

    def open_dialog(self):
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
        self.setWindowTitle(f"nmrproc - {Path(path).name}")
        title = fid.title.splitlines()[0] if fid.title else ""
        self.info_label.setText(
            f"{title}\n{fid.nucleus}, {fid.bf_mhz:.2f} MHz, "
            f"{fid.sw_hz / fid.bf_mhz:.1f} ppm wide, {len(fid.data)} complex points".strip()
        )
        self.ax.set_xlabel(f"Chemical shift, {fid.nucleus} (ppm)" if fid.nucleus else
                           "Chemical shift (ppm)")
        self._set_controls_enabled(True)
        self._sync_controls()
        self.refresh(reset_view=True)
        self.statusBar().showMessage(f"Loaded {path}")
        return True

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.load_path(urls[0].toLocalFile())

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
        self._update_phase_label()
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
        self.ax.set_ylim(bottom - 0.05 * span, top + 0.25 * span)  # headroom for peak labels

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

    # ---- plot events -------------------------------------------------------------------

    def _on_xlim_changed(self, _ax):
        if self._spectrum is None:
            return
        if self.autofit_check.isChecked():
            self._fit_height()
        self._draw_peaks()
        self.canvas.draw_idle()

    def _on_resize(self, _event):
        if self._spectrum is not None:
            self._draw_peaks()

    def _on_motion(self, event):
        if event.inaxes is self.ax and event.xdata is not None and self._spectrum is not None:
            hz = event.xdata * self._spectrum.sf_mhz
            self.statusBar().showMessage(f"{event.xdata:.4f} ppm   ({hz:.1f} Hz)")

    def _on_click(self, event):
        if not (self.pivot_check.isChecked() and event.inaxes is self.ax and event.button == 1):
            return
        mode = getattr(self.toolbar, "mode", "")
        if str(getattr(mode, "value", mode)):  # zoom or pan tool active: ignore the click
            return
        spec = self._spectrum
        index = abs(spec.ppm - event.xdata).argmin()
        self.session.set_pivot(index / (len(spec.ppm) - 1))
        self._sync_controls()
        self.refresh()

    # ---- control handlers --------------------------------------------------------------

    def _on_lb(self, value):
        self.session.settings.lb_hz = value
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
        for widget in (self.lb_spin, self.zf_combo, self.p0_slider, self.p1_slider,
                       self.auto_phase_button, self.reset_phase_button, self.pivot_check,
                       self.baseline_check, self.order_spin, self.peaks_check, self.snr_spin,
                       self.autofit_check, self.reset_view_button):
            widget.setEnabled(enabled)

    def _sync_controls(self):
        """Show the session's settings in the controls without triggering their handlers."""
        s = self.session.settings
        widgets = (self.lb_spin, self.zf_combo, self.order_spin, self.snr_spin,
                   self.baseline_check, self.peaks_check)
        for w in widgets:
            w.blockSignals(True)
        self.lb_spin.setValue(s.lb_hz)
        self.zf_combo.setCurrentIndex(ZERO_FILL_CHOICES.index(s.zero_fill))
        self.order_spin.setValue(s.baseline_order)
        self.snr_spin.setValue(s.peak_snr)
        self.baseline_check.setChecked(s.baseline)
        self.peaks_check.setChecked(s.show_peaks)
        for w in widgets:
            w.blockSignals(False)
        self.p0_slider.set_value(s.p0)
        self.p1_slider.set_value(s.p1)
        if self._spectrum is not None:
            self._update_phase_label()



