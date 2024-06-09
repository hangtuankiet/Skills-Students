import json

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QCheckBox, QLabel
from views import Ui_InputForm, Ui_I_expressionForm, Ui_I_ExpressionRecordForm, Ui_ResultForm
from models import HocSinh


# ======================================================================================================================

import pandas as pd  # Đọc và xử lý dữ liệu.
import numpy as np  # Thao tác dữ liệu số và mảng.
from sklearn.preprocessing import LabelEncoder  # Mã hóa nhãn từ văn bản sang số.
from sklearn.model_selection import train_test_split  # Chia dữ liệu thành tập huấn luyện và kiểm tra.
from tensorflow.keras.models import Sequential  # Mô hình tuần tự.
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # Các lớp mô hình.
from tensorflow.keras.regularizers import l2  # Điều chuẩn trọng số để giảm overfitting.
from tensorflow.keras.optimizers import Adam  # Tối ưu hóa mô hình.
from tensorflow.keras.callbacks import EarlyStopping  # Dừng sớm để tránh overfitting.
from tensorflow.keras.utils import to_categorical  # Chuyển nhãn sang one-hot encoding.
import matplotlib.pyplot as plt  # Vẽ đồ thị để trực quan hóa.
from ipywidgets import widgets  # Tạo giao diện tương tác trong notebook.
from IPython.display import display  # Hiển thị widgets.

data = pd.read_csv('../Data/raw/combination_data.csv')
print("Số lượng hàng và cột:")
print(data.shape)
data.head()

label_encoder = LabelEncoder()

# Cột "Kỹ năng" là nhãn (target) cần dự đoán
X = data.drop('Kỹ năng', axis=1)  # Dữ liệu đầu vào (features)
y = label_encoder.fit_transform(data['Kỹ năng'])  # Dữ liệu đầu ra (target), đã được số hóa

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra với tỉ lệ là 80% huấn luyện và 20% kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Xác định số lượng nhãn duy nhất
num_unique_labels = np.unique(y).shape[0]

# Chuyển đổi nhãn sang one-hot encoding
y_train_one_hot = to_categorical(y_train, num_classes=num_unique_labels)
y_test_one_hot = to_categorical(y_test, num_classes=num_unique_labels)

# Xây dựng mô hình với cấu hình giảm kích thước, tăng L2 regularization và Dropout
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.02)),
    BatchNormalization(),
    Dropout(0.6),
    Dense(16, activation='relu', kernel_regularizer=l2(0.02)),
    BatchNormalization(),
    Dropout(0.6),
    Dense(num_unique_labels, activation='softmax')
])
# Biên dịch mô hình
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Định nghĩa callback EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Huấn luyện mô hình với Early Stopping
history = model.fit(X_train, y_train_one_hot, epochs=52, batch_size=128, validation_split=0.2, verbose=1,
                    callbacks=[early_stopping])

# ======================================================================================================================



class IExpressionRecordForm(QWidget):
    def __init__(self, name:str):
        super().__init__()
        self._ui = Ui_I_ExpressionRecordForm()
        self._ui.setupUi(self)

        # ui turning
        self._ui.name_label.setText(name)

        self._ui.name_label.mousePressEvent = self._onNameLabelClicked

    def _onNameLabelClicked(self, event):
        self._ui.checkBox.setChecked(not self._ui.checkBox.isChecked())

    def getCheckedState(self):
        return 1 if self._ui.checkBox.isChecked() else 0

    def getExpresionName(self):
        return self._ui.name_label.text()

#
class I_EpressionForm(QWidget):
    def __init__(self, title: str, expressionList: list[str], parent=None):
        super(I_EpressionForm, self).__init__()
        self.parent = parent
        self._ui = Ui_I_expressionForm()
        self._ui.setupUi(self)

        # ui turning
        self._ui.list_verticalLayout.setAlignment(Qt.AlignTop)
        self._ui.expressionName_label.setText(title)

        self._expressionList = expressionList

        self._setupData()


    def _getCheckbox(self, content: str):
        cb = QCheckBox(content, self)

        return cb

    def _setupData(self):
        for e in self._expressionList:
            self._ui.list_verticalLayout.addWidget(self._getCheckbox(e))


class ResultForm(QWidget):
    def __init__(self, hocSinh:HocSinh, predictionMess:str, suggestMess:str=''):
        super().__init__()
        self._ui = Ui_ResultForm()
        self._ui.setupUi(self)

        self._ui.prediction_label.setText(predictionMess)
        self._ui.name_label.setText(hocSinh.ten)
        self._ui.class_label.setText(hocSinh.lop)

        self._ui.suggest_label.setText(suggestMess)

        self._ui.back_btn.clicked.connect(self._onBackBtnClicked)

    def _onBackBtnClicked(self):
        self.deleteLater()



class InputController(QWidget):
    def __init__(self, parent=None):
        self.parent = parent
        super().__init__()
        self._ui = Ui_InputForm()
        self._ui.setupUi(self)

        self._gridRow = 0
        self._gridCol = 0

        # So cot hien thi toi da
        self._maxGridCol = 2

        self._maxGridRow = 0

        self._cbList = []

        self._loadExpressions()

        # connect
        self._ui.run_btn.clicked.connect(self._predictions)


    def _displayExpressionCheckBox(self, exp:str):
        cb = IExpressionRecordForm(exp)
        self._cbList.append(cb)
        self._ui.gridLayout.addWidget(cb, self._gridRow, self._gridCol)
        self._gridRow += 1
        if self._gridRow>self._maxGridRow:
            self._gridCol += 1
            self._gridRow = 0


    def _loadExpressions(self):
        expressionList = X.columns.tolist()
        self._maxGridRow = (len(expressionList)/self._maxGridCol).__ceil__()
        for e in expressionList:
            self._displayExpressionCheckBox(e)

    def _getExpressionInput(self):
        missing_columns = []
        selected_features = []
        for u in self._cbList:
            if u.getCheckedState():
                selected_features.append(u.getExpresionName())
            else:
                missing_columns.append(u.getExpresionName())

        # Chuẩn bị dữ liệu dự đoán
        input_data = X.loc[:, selected_features].astype(float)
        # missing_columns = set(X_train.columns) - set(input_data.columns)
        for column in missing_columns:
            input_data[column] = 0
        input_data = input_data[X_train.columns]  # Đảm bảo cùng thứ tự cột
        return input_data

    def _getSuggest(self, highest_skill):
        """
        Tạo gợi ý
        :return: chuỗi gợi ý
        """
        with open("./controllers/SkillAdviceStructured.json", 'r', encoding='utf-8') as file:
            file = json.load(file)
            return '\n'.join(file.get(highest_skill))

    def _predictions(self):
        input_data = self._getExpressionInput()
        # Dự đoán và tìm kỹ năng có tỷ lệ % cao nhất
        predictions = model.predict(input_data)
        predicted_classes = np.argmax(predictions, axis=1)
        probabilities = np.max(predictions, axis=1) * 100  # Tính tỷ lệ %

        # Hiển thị kết quả dự đoán với tỷ lệ % cao nhất
        highest_probability_index = np.argmax(probabilities)
        highest_probability = probabilities[highest_probability_index]
        highest_skill = label_encoder.classes_[predicted_classes[highest_probability_index]]

        # tạo gợi ý
        suggest = self._getSuggest(highest_skill)

        hs = HocSinh(self._ui.name_lineEdit.text(), self._ui.class_lineEdit.text())

        # khởi tạo giao diện kết quả
        rf = ResultForm(hs, f"Kỹ năng dự đoán của bạn là: {highest_skill} với {highest_probability:.2f}% tỷ lệ.",suggest)

        self.parent.setMainWidget(rf)









