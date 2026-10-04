from flask import Flask, render_template, request
import joblib
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Đường dẫn file mô hình
MODEL_PATH = "rust_model.pkl"

# Kiểm tra mô hình đã huấn luyện chưa
model_ready = os.path.exists(MODEL_PATH)
error = None
results = []

# Nếu có mô hình thì load
if model_ready:
    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    classes = model_data["classes"]
else:
    model = None
    classes = []

def preprocess_image(file, size=(64, 64)):
    """Chuyển ảnh thành vector để dự đoán"""
    img = Image.open(file).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).flatten()
    return arr

@app.route("/", methods=["GET", "POST"])
def index():
    global error, results
    results = []
    error = None

    if request.method == "POST":
        try:
            if not model_ready:
                error = "Chưa có mô hình huấn luyện. Vui lòng chạy train.py trước."
            else:
                file = request.files["image"]
                arr = preprocess_image(file)
                y_pred = model.predict([arr])[0]

                # Tạo kết quả hiển thị
                results.append({
                    "filename": file.filename,
                    "label": y_pred,
                    "error": None,
                    "probabilities": [
                        {"label": y_pred, "percent": 100}
                    ]
                })
        except Exception as e:
            error = str(e)

    return render_template("index.html",
                           model_ready=model_ready,
                           error=error,
                           results=results)

if __name__ == "__main__":
    app.run(debug=True)
    