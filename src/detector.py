import os
from ultralytics import YOLO


class FractureDetector:
    def __init__(self, weights_path: str):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Файл весов не найден: {weights_path}")
        self.model = YOLO(weights_path)

    def analyze(self, image_path: str) -> dict:
        results = self.model(image_path, verbose=False)[0]

        response = {
            "status": "success",
            "fractures_detected": len(results.boxes),
            "findings": []
        }

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            response["findings"].append({
                "confidence_score": round(conf, 3),
                "bounding_box": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2}
            })

        return response