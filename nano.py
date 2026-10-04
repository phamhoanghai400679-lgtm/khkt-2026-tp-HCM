"""Small Vietnamese web interface for the trained rust image classifier."""

from functools import lru_cache
from io import BytesIO

import joblib
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

from rust_model import CLASSES, CLASS_LABELS, IMAGE_EXTENSIONS, MODEL_PATH, extract_features

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
MAX_FILES_PER_UPLOAD = 10
PROBABILITY_LABELS = {
    "light_rust": "Gỉ nhẹ",
    "moderated_rust": "Gỉ vừa",
    "heavy_rust": "Gỉ nặng",
}


@lru_cache(maxsize=1)
def load_classifier():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Chưa có mô hình. Hãy thêm ảnh vào dataset/train/ và chạy python train.py."
        )
    saved = joblib.load(MODEL_PATH)
    if tuple(saved.get("classes", ())) != CLASSES:
        raise ValueError(
            "Mô hình đã lưu dùng bộ nhãn cũ. Hãy chạy lại python train.py."
        )
    return saved["model"]


@app.route("/")
def home():
    return render_template("index.html", model_ready=MODEL_PATH.exists())


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("images")
    files = [file for file in files if file.filename]
    if not files:
        return render_template(
            "index.html",
            model_ready=MODEL_PATH.exists(),
            error="Bạn chưa chọn ảnh nào để kiểm tra.",
        ), 400

    if len(files) > MAX_FILES_PER_UPLOAD:
        return render_template(
            "index.html",
            model_ready=MODEL_PATH.exists(),
            error=f"Mỗi lượt chỉ chọn tối đa {MAX_FILES_PER_UPLOAD} ảnh.",
        ), 400

    try:
        model = load_classifier()
    except FileNotFoundError as error:
        return render_template(
            "index.html", model_ready=False, error=str(error)
        ), 503
    except ValueError as error:
        return render_template(
            "index.html", model_ready=True, error=str(error)
        ), 503

    results = []
    for file in files:
        filename = file.filename
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix not in IMAGE_EXTENSIONS:
            results.append({
                "filename": filename,
                "error": "Định dạng không hỗ trợ. Chọn JPG, PNG, WEBP hoặc BMP.",
            })
            continue

        try:
            image_bytes = file.read()
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
                image = image.convert("RGB")
        except (OSError, UnidentifiedImageError):
            results.append({
                "filename": filename,
                "error": "Tệp tải lên không phải ảnh hợp lệ.",
            })
            continue

        probabilities = model.predict_proba(extract_features(image).reshape(1, -1))[0]
        best_index = probabilities.argmax()
        class_name = str(model.classes_[best_index])
        probability_by_class = {
            str(name): float(probability)
            for name, probability in zip(model.classes_, probabilities)
        }
        results.append({
            "filename": filename,
            "label": PROBABILITY_LABELS.get(class_name, CLASS_LABELS[class_name]),
            "confidence": round(float(probabilities[best_index]) * 100, 1),
            "probabilities": [
                {
                    "label": PROBABILITY_LABELS.get(class_key, CLASS_LABELS[class_key]),
                    "percent": round(probability_by_class.get(class_key, 0.0) * 100, 1),
                }
                for class_key in CLASSES
            ],
        })

    if not results:
        return render_template(
            "index.html",
            model_ready=True,
            error="Không tìm thấy ảnh hợp lệ trong lượt tải lên.",
        ), 400

    return render_template(
        "index.html", model_ready=True, results=results
    )


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        model_ready=MODEL_PATH.exists(),
        error="Tổng dung lượng ảnh vượt quá giới hạn 40 MB.",
    ), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)