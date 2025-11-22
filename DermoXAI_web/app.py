import json
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import web_infer

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="DermoXAI Web Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


@app.on_event("startup")
def _load_all_models() -> None:
    global MODELS, MANIFEST
    manifest_path = BASE_DIR / "web_manifest.json"
    if not manifest_path.exists():
        print(f"[WARN] web_manifest.json not found at {manifest_path}")
        MANIFEST = {}
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        MANIFEST = json.load(f)

    cls_ckpts = MANIFEST.get("classifiers", {})
    sev_cfg = MANIFEST.get("severity_head", {})
    unet_cfg = MANIFEST.get("unet", {})

    cls_models, sev_model = web_infer.setup_models(
        ckpt_paths=cls_ckpts,
        severity_ckpt=sev_cfg.get("ckpt"),
        severity_backbone=sev_cfg.get("backbone", "convnext_tiny"),
    )

    seg_model = None
    if unet_cfg.get("ckpt"):
        try:
            seg_model = web_infer.load_unet(unet_cfg["ckpt"])
        except Exception as ex:
            print(f"[WARN] Failed to load UNet: {ex}")

    MODELS["cls_models"] = cls_models
    MODELS["sev_model"] = sev_model
    MODELS["seg_model"] = seg_model

    print("[STARTUP] All models loaded.")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    data = await file.read()

    try:
        cls_out = web_infer.predict(
            data,
            MODELS["cls_models"],
            MODELS["sev_model"],
        )

        seg_out = web_infer.segment_and_abcd_from_bytes(
            data,
            MODELS["seg_model"],
        )

        cam_out = web_infer.multi_backbone_cam_from_bytes(
            data,
            MODELS["cls_models"],
        )

        pred_class = cls_out["pred_class"]
        probs = cls_out["probs"]
        severity = cls_out.get("severity")

        pred_full = web_infer.CLASS_FULLNAME.get(pred_class, pred_class)

        # Simple risk level combining class + severity
        risk_level = "low"
        if pred_class in ["mel", "bcc", "akiec"]:
            if severity is None:
                risk_level = "moderate"
            elif severity >= 2:
                risk_level = "high"
            else:
                risk_level = "moderate"

                pred_full = web_infer.CLASS_FULLNAME.get(pred_class, pred_class)

        resp = {
            "probs": probs,
            "pred_class": pred_class,
            "pred_class_full": pred_full,
            "severity": severity,
            "severity_probs": cls_out.get("severity_probs"),
            "risk_level": risk_level,
            "mel_warning": cls_out.get("mel_warning", False),
            "warning_class": cls_out.get("warning_class"),
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
            "cam_b64": cam_out.get("cam_b64"),
            "class_order": web_infer.CLASS_ORDER,
        }

        return JSONResponse(content=resp)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {ex}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
