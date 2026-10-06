from flask import Flask, render_template, request
import base64
from io import BytesIO
from PIL import Image
from rust_model import load_model, predict
import os

app = Flask(__name__)
model = load_model()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/backup", methods=["GET"])
def backup():
    return render_template("index_backup.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    preview_data = None
    probs = None
    max_label = None

    if "file" in request.files:
        file = request.files["file"]
        if file:
            file_bytes = file.read()

            # tạo ảnh preview base64
            image = Image.open(BytesIO(file_bytes)).convert("RGB")
            image = image.resize((224, 224))
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            preview_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # gọi model để phân tích
            file_stream = BytesIO(file_bytes)
            probs, max_label = predict(model, file_stream)

            # chuyển xác suất sang % và làm tròn
            probs = [round(p * 100, 2) for p in probs]

    return render_template("index.html",
                           preview_data=preview_data,
                           probs=probs,
                           max_label=max_label)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
