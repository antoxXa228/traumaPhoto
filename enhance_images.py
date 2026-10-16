import cv2
from pathlib import Path


def apply_clahe_to_dataset(dataset_dir: Path):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    image_paths = list((dataset_dir / "images").rglob("*.*"))

    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue


        enhanced = clahe.apply(img)

        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(str(img_path), enhanced_bgr)




if __name__ == "__main__":
    apply_clahe_to_dataset(Path("dataset_fracatlas"))