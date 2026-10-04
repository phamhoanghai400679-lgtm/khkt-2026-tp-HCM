"""Train a three-level rust image classifier from labeled folders."""

from collections import Counter
import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from rust_model import (
    CLASSES,
    CLASS_LABELS,
    DATA_DIR,
    MODEL_PATH,
)

# Định dạng ảnh hỗ trợ
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

IMG_SIZE = (64, 64)

def extract_features(image, size=IMG_SIZE):
    """Resize ảnh và chuyển thành vector 1D."""
    image = image.convert("RGB").resize(size)
    arr = np.array(image).flatten()
    return arr

def load_dataset():
    features = []
    labels = []

    for class_name in CLASSES:
        class_dir = DATA_DIR / class_name
        image_paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ) if class_dir.exists() else []

        for path in image_paths:
            try:
                with Image.open(path) as image:
                    image.load()
                    feat = extract_features(image)
                    features.append(feat)
                    labels.append(class_name)
            except (OSError, UnidentifiedImageError) as error:
                print(f"Bỏ qua ảnh không đọc được: {path} ({error})")

    return np.array(features), np.array(labels)

def main():
    print("Đang đọc ảnh trong dataset/train/ ...")
    X, y = load_dataset()
    counts = Counter(y)

    missing_or_small = [
        name for name in CLASSES if counts[name] < 4
    ]
    if missing_or_small:
        print("\nChưa đủ ảnh để huấn luyện và kiểm tra.")
        print("Cần ít nhất 4 ảnh hợp lệ cho mỗi loại:")
        for name in missing_or_small:
            print(f"  - {name}: hiện có {counts[name]} ảnh")
        return

    print("\nSố ảnh theo loại:")
    for name in CLASSES:
        print(f"  {CLASS_LABELS[name]}: {counts[name]}")

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    print("\nĐang huấn luyện mô hình...")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(f"\nĐộ chính xác trên nhóm ảnh kiểm tra: {accuracy_score(y_test, predictions):.1%}")
    print("\nBáo cáo theo từng loại:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(CLASSES),
            target_names=[CLASS_LABELS[name] for name in CLASSES],
            zero_division=0,
        )
    )
    print("Ma trận nhầm lẫn (theo thứ tự nhãn ở trên):")
    print(confusion_matrix(y_test, predictions, labels=list(CLASSES)))

    # 🔥 Dòng thêm để chắc chắn lưu mô hình
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": list(CLASSES)}, MODEL_PATH)
    print(f"\nĐã lưu mô hình tại: {MODEL_PATH}")
    print("Kiểm tra ảnh mới bằng: python predict.py duong_dan/anh.jpg")

if __name__ == "__main__":
    main()
