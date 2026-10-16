from ultralytics import YOLO

def main():
    # Загружаем базовую предобученную модель
    model = YOLO("yolov8m.pt")

    # Запуск обучения
    results = model.train(
        data="fracatlas.yaml",
        epochs=60,
        patience=15,         # остановится досрочно, если точность перестанет расти
        imgsz=1024,          # высокое разрешение для распознавания мелких трещин
        batch=4,             # оптимально под 6 ГБ видеопамяти RTX 3060
        device=0,            # обучение на видеокарте
        workers=2,           # стабильная загрузка данных для Windows
        mosaic=0.5,          # умеренная аугментация
        save=True
    )

if __name__ == '__main__':
    main()