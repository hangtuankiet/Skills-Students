from PyQt5.QtWidgets import QMainWindow, QWidget

from views import Ui_MainWindow
from .inputController import InputController


class MainController(QMainWindow):
    def __init__(self):
        super().__init__()
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

        self.show()
        self.setMainWidget(InputController(self))

    def setMainWidget(self, widget:QWidget):
        self._ui.stackedWidget.addWidget(widget)
        self._ui.stackedWidget.setCurrentWidget(widget)







