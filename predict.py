"""Classify one image with the trained rust model."""

import sys

import joblib

from rust_model import CLASSES, CLASS_LABELS, MODEL_PATH, extract_features, open_image


def main():
    if len(sys.argv) != 2:
        print("Cách dùng: python predict.py duong_dan/anh.jpg")
        raise SystemExit(2)

    image_path = sys.argv[1]
    if not MODEL_PATH.exists():
        print("Chưa có mô hình. Hãy đặt ảnh vào dataset/train/ rồi chạy: python train.py")
        raise SystemExit(1)

    try:
        image = open_image(image_path)
    except (OSError, ValueError) as error:
        print(f"Không đọc được ảnh '{image_path}': {error}")
        raise SystemExit(1) from error

    saved = joblib.load(MODEL_PATH)
    if tuple(saved.get("classes", ())) != CLASSES:
        print("Mô hình đã lưu dùng bộ nhãn cũ. Hãy chạy lại: python train.py")
        raise SystemExit(1)
    model = saved["model"]
    features = extract_features(image).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    best_index = probabilities.argmax()
    class_name = model.classes_[best_index]

    print(f"Kết quả: {CLASS_LABELS[class_name]}")
    print(f"Tỉ lệ dự đoán của mô hình: {probabilities[best_index]:.1%}")
    print("Đây là kết quả tham khảo, không phải phép đo kỹ thuật.")


if __name__ == "__main__":
    main()