import joblib
import numpy as np
from PIL import Image

MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

def preprocess_image(image_file):
    # Đọc ảnh, resize về 64x64, chuẩn hóa pixel
    img = Image.open(image_file).convert("RGB")
    img_resized = img.resize((64, 64))
    img_array = np.array(img_resized) / 255.0  # chuẩn hóa
    img_flatten = img_array.flatten().reshape(1, -1)
    return img_flatten

def predict(image_file):
    try:
        img_array = preprocess_image(image_file)
        probs = model.predict_proba(img_array)[0]

        labels = ["Nhẹ", "Trung bình", "Nặng"]
        result = {labels[i]: f"{probs[i]*100:.2f}%" for i in range(len(labels))}
        max_index = np.argmax(probs)
        result["Kết luận"] = f"AI phân tích: {labels[max_index]} ({probs[max_index]*100:.2f}%)"
        return result
    except Exception as e:
        return {
            "Nhẹ": "--",
            "Trung bình": "--",
            "Nặng": "--",
            "Kết luận": f"Lỗi phân tích ảnh: {str(e)}"
        }
