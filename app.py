import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from feature_builder import build_features
from onnx_infer import SignONNX

# -----------------------------
# App Initialization
# -----------------------------
app = FastAPI()

# -----------------------------
# CORS Configuration (PRODUCTION)
# -----------------------------
# Set this in Render:
# FRONTEND_ORIGINS=https://your-app.vercel.app,https://your-custom-domain.com

origins_env = os.getenv("FRONTEND_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# -----------------------------
# Load Model ONCE at startup
# -----------------------------
model = SignONNX(
    "model/sign_model.onnx",
    "model/labels.json",
    "model/mean.npy",
    "model/std.npy"
)

# -----------------------------
# Request Schemas
# -----------------------------
class Point(BaseModel):
    x: float
    y: float
    z: float

class Hand(BaseModel):
    handedness: str
    points: List[Point]

class PredictRequest(BaseModel):
    hands: List[Hand]

# -----------------------------
# Routes
# -----------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    if not req.hands:
        return {
            "label": "",
            "confidence": 0.0
        }

    features = build_features([h.dict() for h in req.hands])
    label, confidence = model.predict(features)

    return {
        "label": label,
        "confidence": confidence
    }

# -----------------------------
# Health Check (Render-friendly)
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok"
    }
