"""Transform sliders with incremental Shift precision and synchronized numbers."""
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QSlider, QDoubleSpinBox, QPushButton, QHBoxLayout, QVBoxLayout, QApplication


class FineSlider(QSlider):
    """Drag deltas, rather than absolute positions, allow Shift mid-drag without jumps."""
    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._drag_x = None
        self._precise_value = 0.
        self.setToolTip('Arrastra para ajustar. Mantén Shift: precisión ×10. Flechas y rueda también admiten Shift.')

    @staticmethod
    def sensitivity(modifiers):
        return .1 if modifiers & Qt.KeyboardModifier.ShiftModifier else 1.

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._drag_x = event.position().x()
        self._precise_value = float(self.value())
        self.setSliderDown(True)
        event.accept()

    def drag_to(self, x, modifiers):
        if self._drag_x is None:
            return
        span = self.maximum()-self.minimum()
        self._precise_value += (x-self._drag_x)*span/max(1, self.width()-16)*self.sensitivity(modifiers)
        self._precise_value = max(self.minimum(), min(self.maximum(), self._precise_value))
        self._drag_x = x
        self.setValue(round(self._precise_value))

    def mouseMoveEvent(self, event):
        if self._drag_x is not None:
            self.drag_to(event.position().x(), event.modifiers())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_x is not None:
            self.drag_to(event.position().x(), event.modifiers())
            self._drag_x = None
            self.setSliderDown(False)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        steps = {Qt.Key.Key_Left: -1, Qt.Key.Key_Down: -1, Qt.Key.Key_Right: 1, Qt.Key.Key_Up: 1,
                 Qt.Key.Key_PageDown: -10, Qt.Key.Key_PageUp: 10}
        if event.key() in steps:
            increment = max(1, round(self.singleStep()*self.sensitivity(event.modifiers())))
            self.setValue(self.value()+steps[event.key()]*increment)
            event.accept()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        increment = max(1, round(self.singleStep()*self.sensitivity(event.modifiers())))
        self.setValue(self.value()+round(event.angleDelta().y()/120)*increment)
        event.accept()


class TransformControl(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, low, high, default, step, span, parent=None):
        super().__init__(parent)
        self.default = default
        self.span = span
        self.spin = QDoubleSpinBox()
        self.spin.setRange(low, high)
        self.spin.setDecimals(3)
        self.spin.setSingleStep(step)
        self.spin.setValue(default)
        self.spin.setMinimumWidth(84)
        self.slider = FineSlider()
        self.slider.setSingleStep(max(10, round(step*1000)))
        self.slider.setPageStep(self.slider.singleStep()*10)
        self.reset_button = QPushButton('↺')
        self.reset_button.setFixedWidth(32)
        self.reset_button.setToolTip(f'Reiniciar este valor a {default:g}')
        self.reset_button.setAccessibleName('Reiniciar este valor')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(self.spin, 1)
        row.addWidget(self.reset_button)
        layout.addLayout(row)
        layout.addWidget(self.slider)
        self.sync_slider()
        self.spin.valueChanged.connect(self.number_changed)
        self.slider.valueChanged.connect(lambda v: self.spin.setValue(v/1000))
        self.reset_button.clicked.connect(lambda: self.setValue(self.default))

    def value(self):
        return self.spin.value()

    def setValue(self, value):
        self.spin.setValue(value)

    def sync_slider(self):
        value = self.spin.value()
        lo, hi = -self.span, self.span
        if self.spin.minimum() >= 0:
            lo, hi = self.spin.minimum(), self.span
        if value < lo or value > hi:
            lo = max(self.spin.minimum(), value-self.span)
            hi = min(self.spin.maximum(), value+self.span)
        self.slider.blockSignals(True)
        self.slider.setRange(round(lo*1000), round(hi*1000))
        self.slider.setValue(round(value*1000))
        self.slider.blockSignals(False)

    def number_changed(self, value):
        # Keep the scale stable while dragging; typed values can extend the range.
        if value*1000 < self.slider.minimum() or value*1000 > self.slider.maximum():
            self.sync_slider()
        else:
            self.slider.blockSignals(True)
            self.slider.setValue(round(value*1000))
            self.slider.blockSignals(False)
        self.valueChanged.emit(value)
