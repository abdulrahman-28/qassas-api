import sys
import os

# Allow importing from the model/ directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..', 'model'))

import json
import base64
import joblib
import numpy as np
import cv2
import torch
from io import BytesIO
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel

from metrics_factory import MetricsFactory
from config import CATEGORIES, FEATURES

MODEL_DIR = os.path.join(BASE_DIR, '..', 'model', 'trained_models', 'model_all')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading model from {MODEL_DIR} on {DEVICE}...")

    with open(os.path.join(MODEL_DIR, 'model_config.json')) as f:
        cfg = json.load(f)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=True,
    ).to(DEVICE)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, MODEL_DIR)
    pipe.set_progress_bar_config(disable=True)

    state['pipe'] = pipe
    state['iforest'] = joblib.load(os.path.join(MODEL_DIR, 'iforest_model.pkl'))
    state['threshold'] = cfg['optimal_threshold']
    state['strength'] = cfg.get('strength', CATEGORIES['all']['strength'])
    state['guidance'] = cfg.get('guidance', CATEGORIES['all']['guidance'])
    state['metrics'] = MetricsFactory(device=DEVICE)

    print("Model ready.")
    yield

    state.clear()


app = FastAPI(title="Qassas Anomaly Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_anomaly_map(orig_img: Image.Image, recon_img: Image.Image) -> Image.Image:
    orig_np = np.array(orig_img).astype(np.float32)
    recon_np = np.array(recon_img).astype(np.float32)

    diff = np.abs(orig_np - recon_np)
    gray_diff = np.mean(diff, axis=2)
    gray_diff = (gray_diff - gray_diff.min()) / (gray_diff.max() - gray_diff.min() + 1e-8)
    gray_diff = (gray_diff * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(gray_diff, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np.array(orig_img), 0.5, heatmap, 0.5, 0)
    return Image.fromarray(overlay)


def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    try:
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")
    img_512 = img.resize((512, 512))

    prompt = "a high quality photo of a perfect product"

    with torch.no_grad():
        recon = state['pipe'](
            prompt=prompt,
            image=img_512,
            strength=state['strength'],
            guidance_scale=state['guidance'],
            num_inference_steps=30,
            generator=torch.Generator(device=DEVICE).manual_seed(999),
        ).images[0]

    scores = state['metrics'].calculate_metrics(img_512, recon)
    feat = np.array([[scores[f] for f in FEATURES]])

    anomaly_score = float(-state['iforest'].decision_function(feat)[0])
    is_anomalous = anomaly_score >= state['threshold']

    heatmap_img = create_anomaly_map(img_512, recon)

    return {
        "is_anomalous": bool(is_anomalous),
        "score": anomaly_score,
        "heatmap": image_to_base64(heatmap_img),
        "reconstructed": image_to_base64(recon),
    }
