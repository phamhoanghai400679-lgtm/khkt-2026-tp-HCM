"""Shared image features and class labels for the rust classifier."""
from pathlib import Path
import numpy as np
from PIL import Image
    
CLASSES = ("light_rust", "moderated_rust", "heavy_rust")
CLASS_LABELS = {
        "light_rust": "light rust / Gỉ nhẹ",
        "moderated_rust": "moderated rust / Gỉ vừa",
        "heavy_rust": "Heavy Rust / Gỉ nặng",
    }
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMAGE_SIZE = (32, 32)
DATA_DIR = Path("dataset") / "rust_data" / "train"
MODEL_PATH = Path("artifacts") / "rust_classifier.joblib"
    
    
def extract_features(image: Image.Image) -> np.ndarray:
        """Turn an image into simple color, spatial, texture, and histogram features."""
        image = image.convert("RGB").resize(IMAGE_SIZE)
        rgb = np.asarray(image, dtype=np.float32) / 255.0
    
        hsv_image = image.convert("HSV")
        hsv = np.asarray(hsv_image, dtype=np.uint8)
        histograms = []
        for channel in range(3):
            histogram, _ = np.histogram(
                hsv[:, :, channel], bins=16, range=(0, 256), density=True
            )
            histograms.extend(histogram.astype(np.float32))
    
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        rust_color_ratio = np.mean(
            ((hue <= 35) | (hue >= 245))
            & (saturation >= 45)
            & (value >= 25)
            & (value <= 210)
        )
    
        color_stats = np.concatenate((rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1))))
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        horizontal_edges = np.abs(np.diff(gray, axis=1))
        vertical_edges = np.abs(np.diff(gray, axis=0))
        texture_stats = np.asarray(
            [
                horizontal_edges.mean(),
                horizontal_edges.std(),
                vertical_edges.mean(),
                vertical_edges.std(),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            (
                rgb.flatten(),
                np.asarray(histograms, dtype=np.float32),
                color_stats.astype(np.float32),
                texture_stats,
                np.asarray([rust_color_ratio], dtype=np.float32),
            )
        )
    
    
def open_image(path: str | Path) -> Image.Image:
    """Open an image safely and return an independent RGB copy."""
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")
