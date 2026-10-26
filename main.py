import sys
import os
from pathlib import Path
if '--self-test' in sys.argv:
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PyQt6.QtWidgets import QApplication
from charmstudio.window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('L4D2 Charm Studio')
    app.setApplicationVersion('1.5.1')
    app.setOrganizationName('CharmStudio')
    root = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent
    if '--self-test' in sys.argv or '--gpu-test' in sys.argv:
        from charmstudio.selftest import run
        sys.exit(run(app, root, gpu='--gpu-test' in sys.argv))
    window = MainWindow(root)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
