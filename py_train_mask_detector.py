# Import các thư viện cần thiết
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# Khởi tạo các hằng số
INIT_LR = 1e-4  # Tốc độ học ban đầu
EPOCHS = 10     # Số lượng epoch
BS = 32         # Kích thước batch

# Đường dẫn thư mục và các nhãn của dữ liệu ảnh
DIRECTORY = r"D:\Project\Face Mask Detection\dataset"
CATEGORIES = ["with_mask", "without_mask"]

print("[INFO] Tải ảnh...")

# Khởi tạo danh sách để lưu trữ dữ liệu và nhãn
data = []
labels = []

# Lặp qua từng nhãn và tải ảnh
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(category)

# Chuyển đổi nhãn thành dạng one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Chuyển đổi dữ liệu và nhãn thành mảng numpy
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, 
                                                  stratify=labels, random_state=40)

# Tạo dữ liệu augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Tải mô hình MobileNetV2 làm mô hình cơ sở
baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224, 224, 3))
)

# Xây dựng phần đầu của mô hình
headModel = baseModel.output
headModel = MaxPooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="sigmoid")(headModel)

# Kết hợp mô hình cơ sở và phần đầu để tạo mô hình cuối cùng
model = Model(inputs=baseModel.input, outputs=headModel)

# Đóng băng các layer của mô hình cơ sở
for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] Biên dịch mô hình")
# Biên dịch mô hình
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Bắt đầu Huấn luyện...")
# Huấn luyện mô hình
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS
)

print("[INFO] Đánh giá mô hình...")
# Dự đoán và đánh giá mô hình
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

# In báo cáo phân loại
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

print("[INFO] Lưu mô hình mask_detector...")
# Lưu mô hình
model.save("mask_detector.model", save_format="h5")

N = EPOCHS
plt.style.use("ggplot")

fig, axs = plt.subplots(2, 1, figsize=(8, 10))

axs[0].plot(np.arange(0, N), H.history["loss"], label="train_loss")
axs[0].plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
axs[0].set_title("Training and Validation Loss")
axs[0].set_xlabel("Epoch #")
axs[0].set_ylabel("Loss")
axs[0].legend(loc="upper right")

axs[1].plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
axs[1].plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
axs[1].set_title("Training and Validation Accuracy")
axs[1].set_xlabel("Epoch #")
axs[1].set_ylabel("Accuracy")
axs[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig("plot.png")
plt.show()
