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

# Đánh giá mô hình trên tập kiểm tra
loss, accuracy = model.evaluate(X_test, y_test_one_hot)
print(f'Test loss: {loss}\nTest accuracy: {accuracy}')

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
