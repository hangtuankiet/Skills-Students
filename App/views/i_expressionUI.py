# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Workspace\python_project\AIPR\views\i_expressionUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_I_expressionForm(object):
    def setupUi(self, I_expressionForm):
        I_expressionForm.setObjectName("I_expressionForm")
        I_expressionForm.resize(662, 558)
        font = QtGui.QFont()
        font.setPointSize(11)
        I_expressionForm.setFont(font)
        self.verticalLayout = QtWidgets.QVBoxLayout(I_expressionForm)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtWidgets.QWidget(I_expressionForm)
        self.widget.setObjectName("widget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.widget_2 = QtWidgets.QWidget(self.widget)
        self.widget_2.setMinimumSize(QtCore.QSize(0, 40))
        self.widget_2.setMaximumSize(QtCore.QSize(16777215, 40))
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.expressionName_label = QtWidgets.QLabel(self.widget_2)
        self.expressionName_label.setText("")
        self.expressionName_label.setObjectName("expressionName_label")
        self.horizontalLayout.addWidget(self.expressionName_label)
        self.verticalLayout_3.addWidget(self.widget_2)
        self.list_verticalLayout = QtWidgets.QVBoxLayout()
        self.list_verticalLayout.setObjectName("list_verticalLayout")
        self.verticalLayout_3.addLayout(self.list_verticalLayout)
        self.verticalLayout.addWidget(self.widget)

        self.retranslateUi(I_expressionForm)
        QtCore.QMetaObject.connectSlotsByName(I_expressionForm)

    def retranslateUi(self, I_expressionForm):
        _translate = QtCore.QCoreApplication.translate
        I_expressionForm.setWindowTitle(_translate("I_expressionForm", "Form"))