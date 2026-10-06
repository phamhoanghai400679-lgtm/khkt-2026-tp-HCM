import joblib
import numpy as np
from PIL import Image

MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

def preprocess_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img_resized = img.resize((64, 64))
    img_array = np.array(img_resized).flatten().reshape(1, -1)
    return img_array

def predict(image_file):
    try:
        img_array = preprocess_image(image_file)
        probs = model.predict_proba(img_array)[0]

        labels = ["Nhẹ", "Trung bình", "Nặng"]

        # Luôn trả về đủ 3 cấp độ
        result = {labels[i]: f"{probs[i]*100:.2f}%" for i in range(len(labels))}

        # Luôn có kết luận
        max_index = np.argmax(probs)
        result["Kết luận"] = f"AI phân tích: {labels[max_index]} ({probs[max_index]*100:.2f}%)"

        return result
    except Exception as e:
        # Nếu có lỗi, vẫn trả về thông báo rõ ràng
        return {"error": f"Lỗi phân tích ảnh: {str(e)}"}
