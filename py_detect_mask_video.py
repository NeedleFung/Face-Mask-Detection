# Import các thư viện cần thiết
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import cv2

# Định nghĩa hàm để nhận diện và dự đoán việc đeo khẩu trang trên mỗi khuôn mặt trong một khung hình
def detect_and_predict_mask(frame, faceNet, maskNet):
    # Lấy chiều cao và chiều rộng của khung hình
    (h, w) = frame.shape[:2]
    # Tạo blob từ khung hình đầu vào
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Đưa blob vào mạng mạng neural để phát hiện khuôn mặt
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Khởi tạo danh sách các khuôn mặt, vị trí của chúng và các dự đoán
    faces = []
    locs = []
    preds = []

    # Duyệt qua các phát hiện được
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Lọc ra các phát hiện có độ tin cậy cao hơn ngưỡng
        if confidence > 0.5:
            # Tính toán hộp giới hạn của khuôn mặt
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Đảm bảo hộp giới hạn không vượt ra khỏi khung hình
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Trích xuất khuôn mặt từ khung hình và chuẩn bị cho việc dự đoán
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # Thêm khuôn mặt, vị trí và các dự đoán vào danh sách tương ứng
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # Nếu có ít nhất một khuôn mặt được tìm thấy
    if len(faces) > 0:
        # Chuyển danh sách khuôn mặt thành mảng numpy
        faces = np.array(faces, dtype="float64")
        # Dự đoán việc đeo khẩu trang trên các khuôn mặt và lưu kết quả
        preds = maskNet.predict(faces, batch_size=32)

    # Trả về vị trí của các khuôn mặt và các dự đoán
    return (locs, preds)


# Đường dẫn đến file prototxt của mô hình phát hiện khuôn mặt và file weights
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
# Đọc mô hình phát hiện khuôn mặt từ các file đã cung cấp
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Tải mô hình đã được huấn luyện để nhận diện khẩu trang
maskNet = load_model("mask_detector_mendeley_dataset.model")

print("[INFO] Bắt đầu streaming...")
# Bắt đầu streaming từ webcam
vs = VideoStream(src=0).start()

# Vòng lặp chính để nhận diện và dự đoán khẩu trang
while True:
    frame = vs.read()
    # Nhận diện và dự đoán khẩu trang trên khung hình hiện tại
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # Duyệt qua các khuôn mặt đã được nhận diện và dự đoán
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # Xác định nhãn của khuôn mặt dựa trên dự đoán
        label = "An toan" if mask > withoutMask else "Nguy co lay nhiem"
        color = (0, 255, 0) if label == "An toan" else (0, 0, 255)

        # Hiển thị nhãn và hộp giới hạn trên khung hình
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Hiển thị khung hình có các nhãn và hộp giới hạn
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Nếu nhấn 'p', thoát khỏi vòng lặp
    if key == ord("p"):
        break

# Dọn dẹp cửa sổ và dừng streaming từ webcam
cv2.destroyAllWindows()
vs.stop()
