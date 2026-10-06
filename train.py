import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_DIR = "dataset/train"
IMG_SIZE = (64, 64)  # resize ảnh về đúng kích thước huấn luyện

def load_data():
    X, y = [], []
    labels = {"light_rust": 0, "moderated_rust": 1, "heavy_rust": 2}
    for label_name, label_idx in labels.items():
        folder = os.path.join(DATA_DIR, label_name)
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize(IMG_SIZE)
                arr = np.array(img).flatten() / 255.0
                X.append(arr)
                y.append(label_idx)
            except Exception as e:
                print("Lỗi đọc ảnh:", path, e)
    return np.array(X), np.array(y)

def main():
    X, y = load_data()
    print("Dữ liệu:", X.shape, y.shape)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    print("✅ Model đã lưu vào model.pkl")

if __name__ == "__main__":
    main()
