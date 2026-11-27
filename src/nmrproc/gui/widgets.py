"""Small reusable Qt widgets."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class SliderSpin(QtWidgets.QWidget):
    """A labelled slider with a spin box beside it; emits valueChanged(float) on user changes."""

    valueChanged = QtCore.Signal(float)

    def __init__(self, label, minimum, maximum, value=0.0, step=0.1, decimals=1, parent=None):
        super().__init__(parent)
        self._scale = 10**decimals
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(round(minimum * self._scale), round(maximum * self._scale))
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setSuffix("°")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel(label))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)
        layout.addLayout(row)
        self.set_value(value)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def value(self) -> float:
        return self.spin.value()

    def set_value(self, value: float, emit: bool = False) -> None:
        """Set the value; by default silently (no signal), e.g. after an automatic change."""
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        self.spin.setValue(value)
        self.slider.setValue(round(self.spin.value() * self._scale))
        for w in (self.slider, self.spin):
            w.blockSignals(False)
        if emit:
            self.valueChanged.emit(self.spin.value())

    def _from_slider(self, raw: int) -> None:
        self.set_value(raw / self._scale, emit=True)

    def _from_spin(self, value: float) -> None:
        self.set_value(value, emit=True)
