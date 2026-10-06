import joblib
import numpy as np
from PIL import Image

# Load model
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized).flatten().reshape(1, -1)
    return img_array

def predict(image_file):
    # AI phân tích ảnh
    img_array = preprocess_image(image_file)
    probs = model.predict_proba(img_array)[0]

    # 3 cấp độ
    labels = ["Nhẹ", "Trung bình", "Nặng"]

    # Tỉ lệ %
    result = {labels[i]: f"{probs[i]*100:.2f}%" for i in range(len(labels))}

    # Cấp độ cao nhất
    max_index = np.argmax(probs)
    result["Kết luận"] = f"AI phân tích: {labels[max_index]} ({probs[max_index]*100:.2f}%)"

    return result
