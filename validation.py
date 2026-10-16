from pathlib import Path
from ultralytics import YOLO


def show_metrics_in_percent():

    weights = sorted(Path("runs/detect").glob("**/weights/best.pt"))[-1]
    model = YOLO(str(weights))


    metrics = model.val(data="fracatlas.yaml", imgsz=1024, device=0)


    map50_pct = metrics.box.map50 * 100
    map_strict_pct = metrics.box.map * 100
    precision_pct = metrics.box.mp * 100
    recall_pct = metrics.box.mr * 100

    print("\n" + "=" * 45)
    print(f"mAP@50 (основная точность):   {map50_pct:.2f}%")
    print(f"mAP@50-95 (строгая точность): {map_strict_pct:.2f}%")
    print(f"Точность (Precision):         {precision_pct:.2f}%")
    print(f"Полнота (Recall):             {recall_pct:.2f}%")
    print("=" * 45)


if __name__ == "__main__":
    show_metrics_in_percent()