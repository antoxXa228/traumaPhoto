from ultralytics import YOLO

def main():
    weights_path = r"runs\detect\models\fracture_detector-3\weights\best.pt"
    model = YOLO(weights_path)

    # single_cls=True объединяет все классы в 1
    metrics = model.val(data='dataset/data.yaml', single_cls=True, device=0)

    print(f"\nИТОГОВАЯ ТОЧНОСТЬ (mAP50): {metrics.box.map50 * 100:.2f}%")
    print(f"СТРОГАЯ ТОЧНОСТЬ (mAP50-95): {metrics.box.map * 100:.2f}%")

if __name__ == '__main__':
    main()