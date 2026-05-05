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
from threading import Lock

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel

from metrics_factory import MetricsFactory
from config import CATEGORIES, FEATURES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

state: dict = {}

# Serializes inference calls so LoRA adapter swaps don't race
_inference_lock = Lock()


def _model_base_dir() -> str:
    # Support both the old single-model override and the new per-category base dir
    override = os.environ.get("MODEL_DIR_OVERRIDE")
    if override:
        # Legacy: a full path to one model dir → treat its parent as the base
        return os.path.dirname(override)
    return os.environ.get(
        "MODEL_DIR_BASE",
        os.path.join(BASE_DIR, "..", "model", "trained_models"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    base_dir = _model_base_dir()
    print(f"Scanning for category models in {base_dir} on {DEVICE}...")

    # Discover every model_{category} directory that has both required files
    found: dict[str, str] = {}
    for cat in CATEGORIES:
        d = os.path.join(base_dir, f"model_{cat}")
        if (
            os.path.isdir(d)
            and os.path.exists(os.path.join(d, "model_config.json"))
            and os.path.exists(os.path.join(d, "iforest_model.pkl"))
        ):
            found[cat] = d

    if not found:
        raise RuntimeError(
            f"No trained models found under {base_dir}. "
            "Expected sub-directories named model_<category> "
            "(e.g. model_bottle, model_capsule, model_all)."
        )
    print(f"  Found: {sorted(found)}")

    # Load the Stable Diffusion base pipeline once — shared across all categories
    torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    print(f"  Loading SD pipeline...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch_dtype,
        safety_checker=None,
        local_files_only=True,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    # Load each category's LoRA as a named adapter on the shared UNet
    cat_models: dict[str, dict] = {}
    peft_unet = None

    for i, (cat, model_dir) in enumerate(found.items()):
        print(f"  Loading LoRA + IForest for '{cat}'...")
        with open(os.path.join(model_dir, "model_config.json")) as f:
            cfg = json.load(f)

        # Fallback chain mirrors test_single.py exactly
        strength = cfg.get("strength", CATEGORIES.get(cat, CATEGORIES["all"])["strength"])
        guidance = cfg.get("guidance", CATEGORIES.get(cat, CATEGORIES["all"])["guidance"])

        cat_models[cat] = {
            "iforest":   joblib.load(os.path.join(model_dir, "iforest_model.pkl")),
            "threshold": float(cfg["optimal_threshold"]),
            "strength":  strength,
            "guidance":  guidance,
        }

        if i == 0:
            peft_unet = PeftModel.from_pretrained(pipe.unet, model_dir, adapter_name=cat)
        else:
            peft_unet.load_adapter(model_dir, adapter_name=cat)

    pipe.unet = peft_unet

    state["pipe"]       = pipe
    state["peft_unet"]  = peft_unet
    state["cat_models"] = cat_models
    state["metrics"]    = MetricsFactory(device=DEVICE)

    print(f"All models ready. Serving: {sorted(cat_models)}")
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
    orig_np   = np.array(orig_img).astype(np.float32)
    recon_np  = np.array(recon_img).astype(np.float32)
    diff      = np.abs(orig_np - recon_np)
    gray_diff = np.mean(diff, axis=2)
    gray_diff = (gray_diff - gray_diff.min()) / (gray_diff.max() - gray_diff.min() + 1e-8)
    gray_diff = (gray_diff * 255).astype(np.uint8)
    heatmap   = cv2.applyColorMap(gray_diff, cv2.COLORMAP_JET)
    heatmap   = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay   = cv2.addWeighted(np.array(orig_img), 0.5, heatmap, 0.5, 0)
    return Image.fromarray(overlay)


def image_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_models": sorted(state.get("cat_models", {}).keys()),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    product_type: str = Form(default="all"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    requested_cat = product_type.lower()
    cat_models    = state["cat_models"]

    # Resolve which model to use: exact match → 'all' fallback → error
    if requested_cat in cat_models:
        cat = requested_cat
    elif "all" in cat_models:
        print(f"[warn] No model for '{requested_cat}', falling back to 'all'")
        cat = "all"
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No model loaded for product_type '{requested_cat}' "
                f"and no 'all' fallback available. "
                f"Loaded models: {sorted(cat_models)}"
            ),
        )

    model_info = cat_models[cat]
    iforest    = model_info["iforest"]
    threshold  = model_info["threshold"]
    strength   = model_info["strength"]
    guidance   = model_info["guidance"]

    # Prompt and patch mode always follow the REQUESTED category so the
    # reconstruction is appropriate for the actual product, even on fallback
    prompt         = f"a high quality photo of a perfect {requested_cat}"
    use_patch_mode = requested_cat in ("capsule", "pill")

    contents = await file.read()
    print(
        f"[1/5] Image received ({len(contents)} bytes), "
        f"product_type={product_type} → model='{cat}'"
    )
    try:
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")
    print(f"[2/5] Image loaded")

    print(
        f"[3/5] Reconstructing on {DEVICE} "
        f"(mode={'patch' if use_patch_mode else 'full'}, "
        f"strength={strength}, guidance={guidance})..."
    )

    with _inference_lock:
        # Activate the correct LoRA before running the pipeline
        state["peft_unet"].set_adapter(cat)

        if use_patch_mode:
            # Resize to 1024×1024, process as four 512×512 patches — same as test_single.py
            img_1024 = img.resize((1024, 1024))
            patches  = [
                img_1024.crop((0,   0,   512,  512)),
                img_1024.crop((512, 0,  1024,  512)),
                img_1024.crop((0,   512, 512, 1024)),
                img_1024.crop((512, 512, 1024, 1024)),
            ]
            patch_metrics_list: list[dict] = []
            recons: list[Image.Image] = []

            for patch in patches:
                with torch.no_grad():
                    recon_patch = state["pipe"](
                        prompt=prompt,
                        image=patch,
                        strength=strength,
                        guidance_scale=guidance,
                        num_inference_steps=30,
                        generator=torch.Generator(device=DEVICE).manual_seed(999),
                    ).images[0]
                patch_metrics_list.append(
                    state["metrics"].calculate_metrics(patch, recon_patch)
                )
                recons.append(recon_patch)

            # Aggregate worst-case across patches — same as test_single.py
            scores = {
                "L1":        max(s["L1"]        for s in patch_metrics_list),
                "L2":        max(s["L2"]        for s in patch_metrics_list),
                "MS_SSIM":   min(s["MS_SSIM"]   for s in patch_metrics_list),
                "LPIPS":     max(s["LPIPS"]     for s in patch_metrics_list),
                "Max_Patch": max(s["Max_Patch"] for s in patch_metrics_list),
            }

            final_recon = Image.new("RGB", (1024, 1024))
            final_recon.paste(recons[0], (0,   0))
            final_recon.paste(recons[1], (512, 0))
            final_recon.paste(recons[2], (0,   512))
            final_recon.paste(recons[3], (512, 512))

            input_for_map      = img_1024
            recon_for_response = final_recon

        else:
            img_512 = img.resize((512, 512))
            with torch.no_grad():
                recon = state["pipe"](
                    prompt=prompt,
                    image=img_512,
                    strength=strength,
                    guidance_scale=guidance,
                    num_inference_steps=30,
                    generator=torch.Generator(device=DEVICE).manual_seed(999),
                ).images[0]
            scores             = state["metrics"].calculate_metrics(img_512, recon)
            input_for_map      = img_512
            recon_for_response = recon

    print(f"[3/5] Reconstruction done")

    print(f"[4/5] Computing metrics...")
    feat = np.array([[scores[feat_name] for feat_name in FEATURES]])
    print(f"[4/5] Metrics: {scores}")

    # Defect coverage: % of pixels exceeding 30% of the peak absolute difference
    orig_np   = np.array(input_for_map).astype(np.float32)
    recon_np  = np.array(recon_for_response).astype(np.float32)
    diff_gray = np.mean(np.abs(orig_np - recon_np), axis=2)
    diff_max  = diff_gray.max()
    coverage  = (
        float((diff_gray > diff_max * 0.3).sum() / diff_gray.size * 100)
        if diff_max > 0 else 0.0
    )

    print(f"[5/5] Running Isolation Forest...")
    raw_score     = float(iforest.decision_function(feat)[0])
    anomaly_score = -raw_score                 # same convention as test_single.py
    is_anomalous  = anomaly_score >= threshold  # same decision rule as test_single.py
    print(
        f"[5/5] Raw={raw_score:.4f} | Threshold={threshold:.4f} | "
        f"Score={anomaly_score:.4f} | Anomalous={is_anomalous} | Coverage={coverage:.1f}%"
    )

    heatmap_img = create_anomaly_map(input_for_map, recon_for_response)

    return {
        "is_anomalous":  bool(is_anomalous),
        "score":         float(anomaly_score),
        "threshold":     float(threshold),
        "coverage":      round(coverage, 2),
        "metrics":       {f: float(scores[f]) for f in FEATURES},
        "heatmap":       image_to_base64(heatmap_img),
        "reconstructed": image_to_base64(recon_for_response),
    }
