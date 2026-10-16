from ultralytics import YOLO


def main():
    # YOLO11m
    model = YOLO("yolo11m.pt")

    results = model.train(
        data="fracatlas.yaml",
        epochs=100,
        patience=25,
        imgsz=1024,  # res
        batch=4,  # 3060 rtx
        device=0,
        workers=2,

        optimizer="AdamW",
        lr0=0.0005,
        cos_lr=True,


        degrees=15.0,
        fliplr=0.5,
        mosaic=0.3,
        mixup=0.1,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        save=True
    )


if __name__ == "__main__":
    main()