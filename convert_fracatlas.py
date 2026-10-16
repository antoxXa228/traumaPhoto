import json
import random
import shutil
from pathlib import Path

# Базовые пути к распакованному FracAtlas
FRACATLAS_DIR = Path(r"C:\Users\anton\PycharmProjects\TraumaBeta\FracAtlas")
COCO_DIR = FRACATLAS_DIR / "Annotations" / "COCO JSON"
IMAGES_DIR = FRACATLAS_DIR / "images" / "Fractured"

OUTPUT_DIR = Path("dataset_fracatlas")

# Ищем нужный json с Bounding Box (детекция)
target_json = None
for jf in COCO_DIR.glob("*.json"):
    if "box" in jf.name.lower() or "bounding" in jf.name.lower():
        target_json = jf
        break

if not target_json:
    # Если точного совпадения по имени нет, берем первый доступный JSON
    all_jsons = list(COCO_DIR.glob("*.json"))
    if not all_jsons:
        raise FileNotFoundError(f"В папке {COCO_DIR} не найдено ни одного .json файла")
    target_json = all_jsons[0]

print(f"Используется файл разметки: {target_json}")
print(f"Папка со снимками: {IMAGES_DIR}")

with open(target_json, "r", encoding="utf-8") as f:
    coco = json.load(f)

images_info = {img["id"]: img for img in coco["images"]}

bboxes_per_image = {}
for ann in coco["annotations"]:
    img_id = ann["image_id"]
    bboxes_per_image.setdefault(img_id, []).append(ann["bbox"])

img_ids = list(bboxes_per_image.keys())
print(f"Найдено размеченных снимков с переломами: {len(img_ids)}")

# Разделение выборки: 85% train, 15% val
random.seed(42)
random.shuffle(img_ids)
split_idx = int(len(img_ids) * 0.85)
train_ids = img_ids[:split_idx]
val_ids = img_ids[split_idx:]

for split_name, ids in [("train", train_ids), ("val", val_ids)]:
    split_img_dir = OUTPUT_DIR / "images" / split_name
    split_lbl_dir = OUTPUT_DIR / "labels" / split_name
    split_img_dir.mkdir(parents=True, exist_ok=True)
    split_lbl_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_id in ids:
        info = images_info[img_id]
        file_name = info["file_name"]
        w_img, h_img = info["width"], info["height"]

        src_img = IMAGES_DIR / file_name
        if not src_img.exists():
            # Поиск если файл лежит глубже
            matches = list(FRACATLAS_DIR.rglob(file_name))
            if matches:
                src_img = matches[0]
            else:
                continue

        # Копирование снимка
        shutil.copy(src_img, split_img_dir / file_name)

        # Создание файла разметки YOLO
        txt_path = split_lbl_dir / f"{Path(file_name).stem}.txt"
        with open(txt_path, "w", encoding="utf-8") as lf:
            for bbox in bboxes_per_image[img_id]:
                x_min, y_min, bw, bh = bbox
                x_c = (x_min + bw / 2.0) / w_img
                y_c = (y_min + bh / 2.0) / h_img
                norm_w = bw / w_img
                norm_h = bh / h_img
                lf.write(f"0 {x_c:.6f} {y_c:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        copied += 1
    print(f"Сформирована выборка {split_name}: {copied} изображений")

print(f"\nГотово! Датасет сформирован в {OUTPUT_DIR.resolve()}")