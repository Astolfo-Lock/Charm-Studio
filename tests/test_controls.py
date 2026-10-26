import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtTest import QTest, QSignalSpy
from PyQt6.QtWidgets import QApplication
from charmstudio.controls import TransformControl


class TransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.control = TransformControl(-10000,10000,0,.05,10)
        self.control.resize(240,80)
        self.control.show()
        self.app.processEvents()

    def tearDown(self):
        self.control.close()

    def test_shift_drag_ten_times_finer_and_switch_has_no_jump(self):
        slider = self.control.slider
        QTest.mousePress(slider, Qt.MouseButton.LeftButton, pos=QPoint(60,10))
        slider.drag_to(80, Qt.KeyboardModifier.NoModifier)
        regular = self.control.value()
        slider.drag_to(80, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(self.control.value(), regular)
        slider.drag_to(100, Qt.KeyboardModifier.ShiftModifier)
        fine_delta = self.control.value()-regular
        self.assertAlmostEqual(fine_delta, regular/10, delta=.002)
        QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, QPoint(100,10))

    def test_sync_reset_and_single_signal(self):
        spy = QSignalSpy(self.control.valueChanged)
        self.control.setValue(25.75)
        self.assertEqual(self.control.slider.value(),25750)
        self.assertEqual(len(spy),1)
        self.control.slider.setValue(26000)
        self.assertEqual(self.control.value(),26)
        QTest.mouseClick(self.control.reset_button, Qt.MouseButton.LeftButton)
        self.assertEqual(self.control.value(),0)
        self.assertEqual(self.control.slider.value(),0)

    def test_shift_keyboard_and_scale_default(self):
        slider = self.control.slider
        QTest.keyClick(slider, Qt.Key.Key_Right)
        self.assertEqual(self.control.value(),.05)
        QTest.keyClick(slider, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(self.control.value(),.055)
        scale = TransformControl(.001,10,1,.01,10)
        scale.setValue(.4)
        scale.reset_button.click()
        self.assertEqual(scale.value(),1)


if __name__ == '__main__':
    unittest.main()
