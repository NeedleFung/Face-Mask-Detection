# Import các thư viện cần thiết
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# Kích thước ảnh đầu vào
INPUT_SIZE = (224, 224)

# Tải mô hình đã huấn luyện từ file
model = load_model("mask_detector_original_dataset.model")

# Số lượng ảnh kiểm tra
num_test = 300
predict_withmask = 0  # Số lượng dự đoán đeo khẩu trang
actual_withmask = 0   # Số lượng đeo khẩu trang thực tế
predict_withoutmask = 0  # Số lượng dự đoán không đeo khẩu trang
actual_withoutmask = 0   # Số lượng không đeo khẩu trang thực tế
Choices = ["with_mask", "without_mask"]  # Các lựa chọn
Incorrect = []  # Danh sách các dự đoán sai
Name = [[] for _ in range(len(Choices))]  # Danh sách tên các ảnh trong từng lựa chọn

# Lặp qua mỗi lựa chọn và tải tên của các ảnh
for i in range(len(Choices)):
    Name[i] = os.listdir(os.path.expanduser('./dataset_for_testing/' + Choices[i]))

# Duyệt qua số lượng ảnh kiểm tra
for _ in range(num_test):
    # Chọn ngẫu nhiên một lựa chọn
    choice = np.random.randint(0, len(Choices))
    if choice == 0: actual_withmask += 1
    else: actual_withoutmask += 1
    # Chọn một ảnh ngẫu nhiên từ lựa chọn được chọn
    img_id = np.random.randint(0, len(Name[choice]))
    img_path = './dataset_for_testing/' + Choices[choice] + '/' + Name[choice][img_id]

    # Tải ảnh và chuẩn bị cho việc dự đoán
    load_image = image.load_img(img_path, target_size=INPUT_SIZE)
    img_array = image.img_to_array(load_image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Dự đoán nhãn của ảnh
    result = model.predict(img_array)
    prediction = np.argmax(result, axis=1)[0]

    # Chuyển nhãn dự đoán sang chuỗi
    predicted_label = "with_mask" if prediction == 0 else "without_mask"

    # Kiểm tra xem dự đoán có chính xác không và cập nhật số lượng dự đoán đúng
    is_correct = predicted_label == Choices[choice]
    if choice == 0: predict_withmask += is_correct
    else: predict_withoutmask += is_correct

    # Nếu dự đoán sai, thêm thông tin ảnh vào danh sách các dự đoán sai
    if not is_correct:
        Incorrect.append((img_path, Choices[choice], predicted_label))

    print(f"Case {_}: {Choices[choice]} {Name[choice][img_id]} -> {'Đúng' if is_correct else 'Sai'}")

# In ra tỉ lệ dự đoán đúng và số lượng dự đoán sai
correct = predict_withmask + predict_withoutmask
incorrect = num_test - correct
print(f"Đúng: {correct} / {num_test}")
print(f"Sai: {incorrect}")
print(f"Dự đoán đúng đeo khẩu trang: {predict_withmask} / {actual_withmask}")
print(f"Dự đoán đúng không đeo khẩu trang: {predict_withoutmask} / {actual_withoutmask}")
print(f"Độ chính xác: {correct / num_test * 100:.2f}%")
print(f"Độ chính xác của đeo khẩu trang: {predict_withmask / actual_withmask * 100:.2f}%")
print(f"Độ chính xác của không đeo khẩu trang: {predict_withoutmask / actual_withoutmask * 100:.2f}%")

categories = ['Đeo khẩu trang', 'Không đeo khẩu trang']

# Tạo biểu đồ
fig = go.Figure()

# Thêm các cột vào biểu đồ
fig.add_trace(go.Bar(
    x=categories,
    y=[actual_withmask, actual_withoutmask],
    name='Thực tế',
    text=[actual_withmask, actual_withoutmask],
    textposition='auto',
    marker_color='rgba(50, 171, 96, 0.6)'
))

fig.add_trace(go.Bar(
    x=categories,
    y=[predict_withmask, predict_withoutmask],
    name='Dự đoán',
    text=[predict_withmask, predict_withoutmask],
    textposition='auto',
    marker_color='rgba(25, 100, 150, 0.6)'
))

# Cấu hình layout của biểu đồ
fig.update_layout(
    barmode='stack',  # Chế độ xếp chồng
    title='Biểu đồ Tổng quan kết quả của mô hình với 400 ảnh ngẫu nhiên',
    xaxis=dict(title='Kết quả'),
    yaxis=dict(title='Số lượng'),
    legend=dict(title='Chú thích', orientation='v', yanchor='bottom', y=1.02, xanchor='right', x=1),
    margin=dict(l=50, r=50, t=160, b=80),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)

# Hiển thị biểu đồ
fig.write_image("chart.png")

# Hiển thị các ảnh dự đoán sai
for img_info in Incorrect:
    img_path, actual_label, predicted_label = img_info
    img = image.load_img(img_path, target_size=INPUT_SIZE)
    plt.imshow(img)
    plt.title(f"Thực tế: {actual_label}, Dự đoán: {predicted_label}")
    plt.axis('off')
    plt.show()

# %%
