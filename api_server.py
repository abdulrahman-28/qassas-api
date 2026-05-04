import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import json
import base64
import joblib
import numpy as np
import cv2
import torch
from io import BytesIO
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel

from metrics_factory import MetricsFactory
from config import CATEGORIES, FEATURES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL_DIR = os.environ.get(
        "MODEL_DIR_OVERRIDE",
        os.path.join(BASE_DIR, '..', 'model', 'trained_models', 'model_all')
    )
    print(f"Loading model from {MODEL_DIR} on {DEVICE}...")

    with open(os.path.join(MODEL_DIR, 'model_config.json')) as f:
        cfg = json.load(f)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
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

_allow_origin = os.environ.get("ALLOW_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_allow_origin],
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
async def predict(
    file: UploadFile = File(...),
    product_type: str = Form(default="all"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    category = CATEGORIES.get(product_type.lower(), CATEGORIES["all"])
    strength = category["strength"]
    guidance = category["guidance"]

    contents = await file.read()
    print(f"[1/5] Image received ({len(contents)} bytes), product_type={product_type}")
    try:
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")
    img_512 = img.resize((512, 512))
    print(f"[2/5] Image loaded and resized to 512x512")

    prompt = "a high quality photo of a perfect product"

    print(f"[3/5] Running diffusion reconstruction on {DEVICE}... (strength={strength}, guidance={guidance})")
    with torch.no_grad():
        recon = state['pipe'](
            prompt=prompt,
            image=img_512,
            strength=strength,
            guidance_scale=guidance,
            num_inference_steps=30,
            generator=torch.Generator(device=DEVICE).manual_seed(999),
        ).images[0]
    print(f"[3/5] Reconstruction done")

    print(f"[4/5] Computing metrics...")
    scores = state['metrics'].calculate_metrics(img_512, recon)
    feat = np.array([[scores[f] for f in FEATURES]])
    print(f"[4/5] Metrics: {scores}")

    # Compute defect coverage from the pixel-level diff
    orig_np = np.array(img_512).astype(np.float32)
    recon_np = np.array(recon).astype(np.float32)
    diff_gray = np.mean(np.abs(orig_np - recon_np), axis=2)
    diff_max = diff_gray.max()
    coverage = float((diff_gray > diff_max * 0.3).sum() / diff_gray.size * 100) if diff_max > 0 else 0.0

    print(f"[5/5] Running Isolation Forest...")
    raw_score = float(state['iforest'].decision_function(feat)[0])
    anomaly_score = -raw_score  # positive = more anomalous
    is_anomalous = raw_score <= state['threshold']
    print(f"[5/5] Raw: {raw_score:.4f} | Threshold: {state['threshold']:.4f} | Score: {anomaly_score:.4f} | Anomalous: {is_anomalous} | Coverage: {coverage:.1f}%")

    heatmap_img = create_anomaly_map(img_512, recon)

    return {
        "is_anomalous": bool(is_anomalous),
        "score": float(anomaly_score),
        "threshold": float(-state['threshold']),
        "coverage": round(coverage, 2),
        "metrics": {f: float(scores[f]) for f in FEATURES},
        "heatmap": image_to_base64(heatmap_img),
        "reconstructed": image_to_base64(recon),
    }
