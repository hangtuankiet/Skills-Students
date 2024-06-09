import sys

from PyQt5.QtWidgets import QApplication, QMainWindow
from controllers import MainController


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainController()

    sys.exit(app.exec_())

