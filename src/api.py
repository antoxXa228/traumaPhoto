import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.detector import FractureDetector

app = FastAPI(title="TraumaBeta Clinic API")

WEIGHTS_PATH = "models/fracture_detector/weights/best.pt"
detector = None


@app.on_event("startup")
def load_model():
    global detector
    if os.path.exists(WEIGHTS_PATH):
        detector = FractureDetector(WEIGHTS_PATH)
    else:
        print("Внимание: Веса модели еще не обучены!")


@app.post("/api/v1/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    if detector is None:
        raise HTTPException(status_code=500, detail="Модель не обучена или веса не найдены.")

    os.makedirs("temp_uploads", exist_ok=True)
    temp_path = os.path.join("temp_uploads", file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return detector.analyze(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)