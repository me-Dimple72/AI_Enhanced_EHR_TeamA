import cv2
import numpy as np
import os
import sys
from pathlib import Path


def load_grayscale(image_path: Path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Image not readable: {image_path}")
    return img


def apply_denoising(img):
    return cv2.fastNlMeansDenoising(img, h=10)


def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def sharpen_image(img):
    blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    sharp = cv2.addWeighted(img, 1.6, blur, -0.6, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def enhance_pipeline(img):
    img = apply_denoising(img)
    img = apply_clahe(img)
    img = sharpen_image(img)
    return img


def side_by_side(original, enhanced):
    o = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    e = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return np.hstack((o, e))


def save_image(folder: Path, name: str, img):
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / name), img)


def process_dataset(input_dir: Path, output_dir: Path):
    image_files = list(input_dir.glob("*.[jJ][pP]*g")) + list(input_dir.glob("*.[pP][nN]g"))

    if not image_files:
        print("🚨 No images found.")
        return

    for img_path in image_files:
        try:
            original = load_grayscale(img_path)
            enhanced = enhance_pipeline(original)
            comparison = side_by_side(original, enhanced)

            save_image(output_dir / "originals", img_path.name, original)
            save_image(output_dir / "enhanced", img_path.name, enhanced)
            save_image(
                output_dir / "comparisons",
                img_path.stem + "_compare.jpg",
                comparison,
            )

            print(f"✅ Processed: {img_path.name}")

        except Exception as err:
            print(f"❌ Failed: {img_path.name} | {err}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python xray_batch_processor.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = Path(sys.argv[1])
    output_folder = Path(sys.argv[2])

    process_dataset(input_folder, output_folder)
