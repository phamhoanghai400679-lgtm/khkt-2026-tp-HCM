from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import joblib
import os

app = Flask(__name__)

# --- Load model tại đây ---
MODEL_PATH = "model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Không tìm thấy file {MODEL_PATH}")

@app.route("/")
def index():

    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")

        # xử lý ảnh
        img_resized = img.resize((64, 64))
        img_array = np.array(img_resized).flatten().reshape(1, -1)

        # dự đoán xác suất cho 3 lớp
        probs = model.predict_proba(img_array)[0]

        result = {
            "class_1": f"{probs[0]*100:.2f}%",
            "class_2": f"{probs[1]*100:.2f}%",
            "class_3": f"{probs[2]*100:.2f}%"
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
