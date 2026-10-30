import os
import torch

# --- PERMANENT PYTORCH 2.6 UNPICKLING BYPASS ---
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
# ----------------------------------------------

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
import io

app = FastAPI(title="Oasyra Food ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
YOLO_PATH = os.path.join(BASE_DIR, "models", "best.pt")
MOBILENET_PATH = os.path.join(BASE_DIR, "models", "mobilenetv3_small_sih_best.pth")

yolo_model = None
mobilenet_model = None

@app.on_event("startup")
def load_models():
    global yolo_model, mobilenet_model
    
    try:
        print("Loading YOLO11n model...")
        yolo_model = YOLO(YOLO_PATH)
        print("YOLO11n loaded successfully.")
    except Exception as e:
        print(f"Failed to load YOLO11n: {e}")

    try:
        print("Loading MobileNetV3-Small model...")
        mobilenet_model = torch.load(MOBILENET_PATH, map_location=torch.device('cpu'))
        print("MobileNetV3-Small loaded successfully.")
    except Exception as e:
        print(f"Failed to load MobileNetV3-Small: {e}")

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "service": "Oasyra ML Inference API",
        "yolo_loaded": yolo_model is not None,
        "mobilenet_loaded": mobilenet_model is not None
    }

@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Send an image.")
    
    if yolo_model is None:
        raise HTTPException(status_code=500, detail="YOLO model not loaded.")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = yolo_model(image)
    detected_boxes = []
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = yolo_model.names[cls_id]
            detected_boxes.append({
                "class": name, 
                "confidence": conf,
                "box": box.xyxy[0].tolist()
            })

    return {
        "success": True,
        "detections": detected_boxes
    }