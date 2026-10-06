import joblib
import numpy as np
from PIL import Image

# Load model khi import module
MODEL_PATH = "rust_model.pkl"
model = joblib.load(MODEL_PATH)

def preprocess_image(image_file):
    """
    Nhận file ảnh (stream từ Flask), chuyển thành vector phù hợp với model.
    """
    img = Image.open(image_file).convert("RGB")
    img_resized = img.resize((128, 128))  # resize cố định
    img_array = np.array(img_resized).flatten().reshape(1, -1)
    return img_array

def predict(image_file):
    """
    Trả về xác suất (%) cho 3 cấp độ: Nhẹ, Trung bình, Nặng.
    """
    img_array = preprocess_image(image_file)
    probs = model.predict_proba(img_array)[0]  # [p1, p2, p3]
    return {
        "Nhẹ": f"{probs[0]*100:.2f}%",
        "Trung bình": f"{probs[1]*100:.2f}%",
        "Nặng": f"{probs[2]*100:.2f}%"
    }
