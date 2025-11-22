import os
import json
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

import web_infer
from openai import OpenAI

# ===============================
# Global variables
# ===============================

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="DermoXAI Web Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Render requires this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS: Dict[str, Any] = {
    "cls_models": None,
    "sev_model": None,
    "seg_model": None,
}

MANIFEST: Dict[str, Any] = {}

# ===============================
# Utility: Download weights if missing
# ===============================

def download_if_missing(path: Path, url: str) -> None:
    """
    Download .pth file from Google Drive if it does not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"[OK] Weight exists: {path}")
        return

    print(f"[DOWNLOAD] Downloading {path.name} from {url} ...")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"[DONE] Downloaded {path.name} → {path}")


# ===============================
# SkinGPT-lite (OpenAI API)
# ===============================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None

if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY NOT SET → /api/chat will not work.")
else:
    client = OpenAI(api_key=OPENAI_API_KEY)


class ChatTurn(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatTurn]] = None


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    if client is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Chat backend not configured (missing OPENAI_API_KEY)."},
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are DermAI-Lite, a dermatology education assistant. "
                "Explain skin lesions, ABCD rules, melanoma warning signs, "
                "risk factors, and when to see a dermatologist. "
                "NEVER give a diagnosis or prescribe medicine. "
                "Be concise, structured and safe."
            ),
        }
    ]

    if req.history:
        for turn in req.history[-10:]:
            if turn.role in {"user", "assistant"}:
                messages.append(
                    {"role": turn.role, "content": turn.content}
                )

    messages.append({"role": "user", "content": req.message})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )
        reply = resp.choices[0].message.content
        return {"reply": reply}

    except Exception as e:
        print("[ERROR] Chat error:", e)
        return JSONResponse(
            status_code=500,
            content={"error": "Chat backend error. Try again later."},
        )


# ===============================
# Startup: Download + Load all models
# ===============================

@app.on_event("startup")
def load_models_on_startup():
    """Load all classification + severity + U-Net models, downloading if needed."""
    global MODELS, MANIFEST

    manifest_path = BASE_DIR / "web_manifest.json"
    if not manifest_path.exists():
        print("[WARN] web_manifest.json NOT FOUND")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        MANIFEST = json.load(f)

    # -------- Download classifier weights
    cls_ckpts = MANIFEST.get("classifiers", {})
    download_urls = MANIFEST.get("download_urls", {})

    for name, rel_path in cls_ckpts.items():
        url = download_urls.get(name)
        if url:
            try:
                download_if_missing(BASE_DIR / rel_path, url)
            except Exception as e:
                print(f"[WARN] Failed to download {name}: {e}")

    # -------- Download severity head
    sev_cfg = MANIFEST.get("severity_head", {})
    if "ckpt" in sev_cfg and "url" in sev_cfg:
        try:
            download_if_missing(BASE_DIR / sev_cfg["ckpt"], sev_cfg["url"])
        except Exception as e:
            print(f"[WARN] Failed to download severity head: {e}")

    # -------- Download UNet
    unet_cfg = MANIFEST.get("unet", {})
    if "ckpt" in unet_cfg and "url" in unet_cfg:
        try:
            download_if_missing(BASE_DIR / unet_cfg["ckpt"], unet_cfg["url"])
        except Exception as e:
            print(f"[WARN] Failed to download UNet: {e}")

    # -------- Load models normally
    try:
        cls_models, sev_model = web_infer.setup_models(
            ckpt_paths=cls_ckpts,
            severity_ckpt=sev_cfg.get("ckpt"),
            severity_backbone=sev_cfg.get("backbone", "convnext_tiny"),
        )
    except Exception as e:
        print("[FATAL] Failed to load classifier models:", e)
        raise

    seg_model = None
    try:
        if unet_cfg.get("ckpt"):
            seg_model = web_infer.load_unet(unet_cfg["ckpt"])
    except Exception as e:
        print("[WARN] Failed to load UNet:", e)

    MODELS["cls_models"] = cls_models
    MODELS["sev_model"] = sev_model
    MODELS["seg_model"] = seg_model

    print("[STARTUP] All models loaded successfully.")


# ===============================
# Routes
# ===============================

@app.get("/")
def root():
    """Serve frontend HTML."""
    return FileResponse(BASE_DIR / "index.html")


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image.")

    raw = await file.read()

    try:
        cls_out = web_infer.predict(raw, MODELS["cls_models"], MODELS["sev_model"])
        seg_out = web_infer.segment_and_abcd_from_bytes(raw, MODELS["seg_model"])
        cam_out = web_infer.multi_backbone_cam_from_bytes(raw, MODELS["cls_models"])

        pred_class = cls_out["pred_class"]
        probs = cls_out["probs"]
        severity = cls_out.get("severity")

        # Severity → risk category
        risk = None
        if severity == 0:
            risk = "low"
        elif severity == 1:
            risk = "moderate"
        elif severity == 2:
            risk = "high"

        # Upgrade if mel_warning
        if cls_out.get("mel_warning") and risk != "high":
            risk = "high"

        pred_full = web_infer.CLASS_FULLNAME.get(pred_class, pred_class)

        resp = {
            "probs": probs,
            "pred_class": pred_class,
            "pred_class_full": pred_full,
            "severity": severity,
            "severity_probs": cls_out.get("severity_probs"),
            "risk_level": risk,
            "mel_warning": cls_out.get("mel_warning"),
            "warning_class": cls_out.get("warning_class"),

            # segmentation
            "mask_overlay_b64": seg_out.get("mask_overlay_b64"),
            "abcd": seg_out.get("abcd_explanation", {}),
            "abcd_raw": {
                "A_asymmetry": seg_out.get("A_asymmetry"),
                "B_border_irreg": seg_out.get("B_border_irreg"),
                "C_clusters": seg_out.get("C_clusters"),
                "C_color_std": seg_out.get("C_color_std"),
                "D_area_ratio": seg_out.get("D_area_ratio"),
                "D_equiv_diam": seg_out.get("D_equiv_diam"),
                "mask_valid": seg_out.get("mask_valid"),
            },

            # CAM
            "cam_b64": cam_out.get("cam_b64"),

            "class_order": web_infer.CLASS_ORDER,
        }

        return JSONResponse(resp)

    except Exception as e:
        print("[ERROR] Prediction failed:", e)
        raise HTTPException(500, f"Error during prediction: {e}")


# ===============================
# Local dev run
# ===============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
