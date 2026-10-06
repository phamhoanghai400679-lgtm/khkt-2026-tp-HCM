import joblib
import numpy as np
from PIL import Image
from io import BytesIO

# Kích thước ảnh phải khớp với lúc huấn luyện
IMG_SIZE = (64, 64)

def load_model():
    """Load model đã huấn luyện từ file model.pkl"""
    model = joblib.load("model.pkl")
    return model

def predict(model, file_bytes):
    """Dự đoán mức độ gỉ sét từ ảnh"""
    # Đọc ảnh từ file upload
    img = Image.open(file_bytes).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).flatten() / 255.0  # chuẩn hóa pixel

    # Dự đoán xác suất cho từng lớp
    probs = model.predict_proba([arr])[0]
    labels = ["Gỉ nhẹ", "Gỉ trung bình", "Gỉ nặng"]

    # Tạo dict kết quả
    result = {labels[i]: float(probs[i]) for i in range(len(labels))}
    max_label = labels[np.argmax(probs)]

    return result, max_label
