from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from feature_builder import build_features
from onnx_infer import SignONNX

app = FastAPI()

model = SignONNX(
    "model/sign_model.onnx",
    "model/labels.json",
    "model/mean.npy",
    "model/std.npy"
)

# -------- Request Schema --------
class Point(BaseModel):
    x: float
    y: float
    z: float

class Hand(BaseModel):
    handedness: str
    points: List[Point]

class PredictRequest(BaseModel):
    hands: List[Hand]

# -------- Endpoint --------
@app.post("/predict")
def predict(req: PredictRequest):
    if not req.hands:
        return { "label": "", "confidence": 0.0 }

    features = build_features([h.dict() for h in req.hands])
    label, confidence = model.predict(features)

    return {
        "label": label,
        "confidence": confidence
    }
