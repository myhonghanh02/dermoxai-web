"""
DermoXAI – Model utilities for web backend.

Features:
- Load 7-class classifiers (DenseNet121, EfficientNet-B0, ConvNeXt-Tiny, Swin-Tiny).
- Ensemble classification (avg softmax over all backbones).
- CORAL-style severity head (0/1/2).
- UNet-ResNet34 segmentation + ABCD morphology metrics.
- Multi-backbone Grad-CAM-like heatmap (average of backbones).
"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# Optional deps
try:
    import timm  # type: ignore
except Exception:
    timm = None

try:
    import segmentation_models_pytorch as smp  # type: ignore
except Exception:
    smp = None


# ---------------------------------------------------------------------
# GLOBAL CONSTANTS
# ---------------------------------------------------------------------
CLASS_ORDER = ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"]

CLASS_FULLNAME = {
    "akiec": "Actinic keratosis / Bowen’s disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Seborrheic keratosis",
    "df": "Dermatofibroma",
    "nv": "Melanocytic nevus",
    "mel": "Melanoma",
    "vasc": "Vascular lesion",
}

IDX2CLASS = {i: c for i, c in enumerate(CLASS_ORDER)}

IMG_SIZE = 224
SEG_IMG_SIZE = 256

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === PRIOR BIAS CORRECTION (tối ưu từ confusion matrix) ===============
# Order: [akiec, bcc, bkl, df, nv, mel, vasc]
# - Giảm mạnh nevus.
# - Boost các lớp ác tính/hiếm (mel, df) và lớp hay bị nuốt (bkl).
CLASS_PRIOR_CORRECTION = torch.tensor(
    [1.22, 1.18, 1.22, 1.30, 0.28, 2.35, 1.12],
    dtype=torch.float32,
    device=DEVICE,
)


# ---------------------------------------------------------------------
# BASIC UTILITIES
# ---------------------------------------------------------------------
def _resolve_ckpt_path(raw: str) -> Path:
    here = Path(__file__).resolve().parent
    candidates = [Path(raw), here / raw, here / "weights" / Path(raw).name]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0 : y0 + side, x0 : x0 + side]


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/png;base64," + b64


# ---------------------------------------------------------------------
# CLASSIFICATION MODELS
# ---------------------------------------------------------------------
def _build_model(name: str, num_classes: int) -> nn.Module:
    norm = name.lower()

    # DenseNet121
    if "densenet" in norm:
        try:
            from torchvision.models import densenet121

            model = densenet121(weights=None)
            in_ch = model.classifier.in_features
            model.classifier = nn.Linear(in_ch, num_classes)
        except Exception:
            assert timm is not None, "Need timm to build densenet121."
            model = timm.create_model("densenet121", pretrained=False, num_classes=num_classes)
        return model

    # EfficientNet-B0
    if "efficientnet" in norm:
        try:
            from torchvision.models import efficientnet_b0

            model = efficientnet_b0(weights=None)
            in_ch = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_ch, num_classes)
        except Exception:
            assert timm is not None, "Need timm to build efficientnet_b0."
            model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
        return model

    # ConvNeXt-Tiny
    if "convnext" in norm:
        try:
            from torchvision.models import convnext_tiny

            model = convnext_tiny(weights=None)
            in_ch = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_ch, num_classes)
        except Exception:
            assert timm is not None, "Need timm to build convnext_tiny."
            model = timm.create_model("convnext_tiny", pretrained=False, num_classes=num_classes)
        return model

    # Swin-Tiny
    if "swin" in norm:
        try:
            from torchvision.models import swin_t

            model = swin_t(weights=None)
            in_ch = model.head.in_features
            model.head = nn.Linear(in_ch, num_classes)
        except Exception:
            assert timm is not None, "Need timm to build swin_tiny."
            model = timm.create_model(
                "swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes
            )
        return model

    raise ValueError(f"Unknown backbone name: {name}")


class SeverityHead(nn.Module):
    """
    Severity head that matches the training setup:
    - ConvNeXt-Tiny backbone (pretrained=True, global_pool='avg', num_classes=0).
    - Linear head for (sev_classes-1) CORAL logits.
    """

    def __init__(self, backbone_name: str = "convnext_tiny", sev_classes: int = 3):
        super().__init__()
        if timm is None:
            raise RuntimeError("timm is required to build SeverityHead.")

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = nn.Linear(self.backbone.num_features, sev_classes - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)


def _cls_tfm() -> T.Compose:
    return T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ]
    )


# ---------------------------------------------------------------------
# MODEL SETUP (called from app.py)
# ---------------------------------------------------------------------
def setup_models(
    ckpt_paths: Dict[str, str],
    severity_ckpt: Optional[str] = None,
    severity_backbone: str = "convnext_tiny",
) -> Tuple[Dict[str, nn.Module], Optional[nn.Module]]:
    models: Dict[str, nn.Module] = {}
    for name, raw_path in ckpt_paths.items():
        ckpt_path = _resolve_ckpt_path(raw_path)
        if not ckpt_path.exists():
            print(f"[WARN] Checkpoint for {name} not found at {ckpt_path}")
            continue
        model = _build_model(name, num_classes=len(CLASS_ORDER)).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        model.eval()
        models[name] = model
        print(f"[INIT] Loaded classifier {name} from {ckpt_path}")

    sev_model: Optional[nn.Module] = None
    if severity_ckpt is not None:
        sev_path = _resolve_ckpt_path(severity_ckpt)
        if sev_path.exists():
            sev_model = SeverityHead(severity_backbone).to(DEVICE)
            sev_state = torch.load(sev_path, map_location=DEVICE)
            sev_model.load_state_dict(sev_state, strict=False)
            sev_model.eval()
            print(f"[INIT] Loaded severity head from {sev_path}")
        else:
            print(f"[WARN] Severity checkpoint not found at {sev_path}")

    return models, sev_model


# ---------------------------------------------------------------------
# ENSEMBLE PREDICTION + SEVERITY
# ---------------------------------------------------------------------
@torch.no_grad()
def predict(
    image_bytes: bytes,
    models: Dict[str, nn.Module],
    sev_model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    if not models:
        raise RuntimeError("No classification models loaded.")

    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_raw = np.array(pil)
    np_cropped = _center_crop_square(np_raw)
    pil_sq = Image.fromarray(np_cropped)

    x = _cls_tfm()(pil_sq).unsqueeze(0).to(DEVICE)

    # --- ENSEMBLE LOGITS ---
    logits_list = []
    for m in models.values():
        m.eval()
        logits_list.append(m(x))

    stacked = torch.stack(logits_list, dim=0)  # (M,1,C)
    avg_logits = stacked.mean(dim=0)[0]

    # --- PRIOR BIAS CORRECTION ---
    if CLASS_PRIOR_CORRECTION is not None:
        logits_adj = avg_logits + torch.log(CLASS_PRIOR_CORRECTION)
    else:
        logits_adj = avg_logits

    probs_tensor = F.softmax(logits_adj, dim=-1)
    probs = probs_tensor.tolist()

    # --- SEVERITY (tính sớm để dùng nếu sau này muốn rule phức tạp hơn) ---
    severity_int: Optional[int] = None
    severity_probs: Optional[List[float]] = None
    if sev_model is not None:
        sev_model.eval()
        sev_logits = sev_model(x)[0]  # (2,) for 3-level CORAL
        sev_probs_t = torch.sigmoid(sev_logits)
        severity_probs = sev_probs_t.tolist()
        severity_int = int((sev_probs_t > 0.5).sum().item())
        severity_int = max(0, min(severity_int, 2))

    # --- TOP-2 WARNING SYSTEM (generic, không chỉ melanoma) -------------
    mel_warning = False    # dùng như flag "has warning"
    warning_class: Optional[str] = None

    nv_idx = CLASS_ORDER.index("nv")
    probs_sorted, idx_sorted = torch.sort(probs_tensor, descending=True)

    top1_idx = int(idx_sorted[0].item())
    top2_idx = int(idx_sorted[1].item())
    top1_p = float(probs_sorted[0].item())
    top2_p = float(probs_sorted[1].item())

    pred_idx = top1_idx
    pred_class = IDX2CLASS[pred_idx]

    # Nếu top-1 là nevus nhưng top-2 cũng khá cao => cảnh báo theo top-2
    # Ngưỡng: top2_p >= 10%
    if top1_idx == nv_idx and top2_p >= 0.10:
        mel_warning = True
        warning_class = IDX2CLASS[top2_idx]

    return {
        "pred_class": pred_class,
        "probs": probs,
        "severity": severity_int,
        "severity_probs": severity_probs,
        "mel_warning": mel_warning,
        "warning_class": warning_class,
    }


# ---------------------------------------------------------------------
# UNET SEGMENTATION + ABCD METRICS
# ---------------------------------------------------------------------
def load_unet(seg_ckpt_path: str) -> nn.Module:
    if smp is None:
        raise RuntimeError("segmentation_models_pytorch is required for UNet.")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    path = _resolve_ckpt_path(seg_ckpt_path)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    print(f"[INIT] Loaded UNet from {path}")
    return model


def _seg_tfm() -> T.Compose:
    return T.Compose(
        [
            T.Resize((SEG_IMG_SIZE, SEG_IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ]
    )


def _compute_asymmetry(mask01: np.ndarray) -> float:
    h, w = mask01.shape
    total = float(mask01.sum() + 1e-8)
    if total <= 1e-6:
        return float("nan")
    left = float(mask01[:, : w // 2].sum())
    right = float(mask01[:, w // 2 :].sum())
    top = float(mask01[: h // 2, :].sum())
    bottom = float(mask01[h // 2 :, :].sum())
    return (abs(left - right) + abs(top - bottom)) / (2.0 * total)


def _compute_border_irregularity(mask01: np.ndarray) -> float:
    cnts, _ = cv2.findContours(mask01.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return float("nan")
    cnt = max(cnts, key=cv2.contourArea)
    A = cv2.contourArea(cnt)
    P = cv2.arcLength(cnt, True)
    if P <= 1e-6 or A <= 0:
        return float("nan")
    circularity = 4.0 * math.pi * A / (P * P + 1e-8)
    return float(max(0.0, 1.0 - circularity))


def _compute_color_features(img_rgb: np.ndarray, mask01: np.ndarray) -> Tuple[float, float]:
    lesion = img_rgb[mask01 > 0]
    if lesion.size < 50:
        return float("nan"), float("nan")

    lab = cv2.cvtColor(lesion.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    K = max(2, min(5, len(lab) // 50))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)
    _, labels, _ = cv2.kmeans(
        lab.astype(np.float32),
        K,
        None,
        criteria,
        5,
        cv2.KMEANS_RANDOM_CENTERS,
    )

    cnt = np.bincount(labels.flatten(), minlength=K)
    frac = cnt / (cnt.sum() + 1e-8)
    clusters = int((frac >= 0.05).sum())
    color_std = float(lab.std(axis=0).mean() / 255.0)
    return float(clusters), color_std


def _compute_diameter_features(mask01: np.ndarray) -> Tuple[float, float]:
    h, w = mask01.shape
    A = float(mask01.sum())
    if A <= 0:
        return float("nan"), float("nan")
    area_ratio = A / float(h * w + 1e-8)
    equiv_d = 2.0 * math.sqrt(A / (math.pi + 1e-8))
    return float(area_ratio), float(equiv_d)


def _sanity_check_mask(
    mask01: np.ndarray,
    min_ratio: float = 0.002,
    max_ratio: float = 0.6,
) -> Tuple[bool, float]:
    h, w = mask01.shape
    r = float(mask01.sum()) / float(h * w + 1e-8)
    return bool(min_ratio <= r <= max_ratio), r


def _explain_abcd(feat: Dict[str, float]) -> Dict[str, str]:
    A = float(feat.get("A_asymmetry", float("nan")))
    B = float(feat.get("B_border_irreg", float("nan")))
    Cc = float(feat.get("C_clusters", float("nan")))
    Da = float(feat.get("D_area_ratio", float("nan")))

    if not math.isfinite(A):
        a_txt = "The lesion’s symmetry could not be measured reliably on this image."
    elif A < 0.12:
        a_txt = "The lesion appears mostly symmetric when compared across opposite halves."
    elif A < 0.25:
        a_txt = "The lesion shows mild asymmetry between its halves."
    else:
        a_txt = "The lesion is clearly asymmetric in shape."

    if not math.isfinite(B):
        b_txt = "The border outline is difficult to estimate, so edge irregularity is reported with caution."
    elif B < 0.12:
        b_txt = "Lesion borders look smooth and well-defined overall."
    elif B < 0.25:
        b_txt = "Lesion borders are slightly uneven with some soft irregularities."
    else:
        b_txt = "Lesion borders appear irregular and jagged in several areas."

    if not math.isfinite(Cc):
        c_txt = "Overall colour appears fairly uniform on this image."
    elif Cc <= 1.5:
        c_txt = "The lesion is mostly one colour tone."
    elif Cc <= 3:
        c_txt = "The lesion contains a few distinct colour tones."
    else:
        c_txt = "The lesion shows multiple colour tones, suggesting heterogeneous pigmentation."

    if not math.isfinite(Da):
        d_txt = "The apparent lesion size cannot be estimated reliably from this mask."
    elif Da < 0.02:
        d_txt = "The segmented lesion occupies a small fraction of the image."
    elif Da < 0.07:
        d_txt = "The lesion covers a moderate portion of the image."
    else:
        d_txt = "The lesion covers a relatively large area of the image."

    return {"A": a_txt, "B": b_txt, "C": c_txt, "D": d_txt}


@torch.no_grad()
def segment_and_abcd_from_bytes(
    image_bytes: bytes,
    seg_model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Run UNet segmentation + ABCD metrics + overlay.

    Overlay rule:
      - Lesion region: keep original colour/brightness.
      - Background (outside lesion): tinted with #314026 at ~40% opacity.
    """
    if seg_model is None:
        return {
            "has_segmentation": False,
            "mask_valid": None,
            "area_ratio": None,
            "A_asymmetry": None,
            "B_border_irreg": None,
            "C_clusters": None,
            "C_color_std": None,
            "D_area_ratio": None,
            "D_equiv_diam": None,
            "abcd_explanation": {},
            "mask_overlay_b64": None,
        }

    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_img = np.array(pil)
    np_sq = _center_crop_square(np_img)
    pil_sq = Image.fromarray(np_sq)

    x_seg = _seg_tfm()(pil_sq).unsqueeze(0).to(DEVICE)
    prob = torch.sigmoid(seg_model(x_seg))[0, 0].cpu().numpy()
    mask01 = (prob > 0.5).astype(np.uint8)
    mask01 = cv2.resize(mask01, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    valid, area_ratio = _sanity_check_mask(mask01)

    if not valid:
        feat = {
            "has_segmentation": True,
            "mask_valid": False,
            "area_ratio": float(area_ratio),
            "A_asymmetry": None,
            "B_border_irreg": None,
            "C_clusters": None,
            "C_color_std": None,
            "D_area_ratio": None,
            "D_equiv_diam": None,
        }
        feat["abcd_explanation"] = _explain_abcd(
            {
                "A_asymmetry": float("nan"),
                "B_border_irreg": float("nan"),
                "C_clusters": float("nan"),
                "D_area_ratio": float("nan"),
            }
        )
        feat["mask_overlay_b64"] = None
        return feat

    # --- Metrics ---
    A = _compute_asymmetry(mask01)
    B = _compute_border_irregularity(mask01)

    rgb_for_color = cv2.resize(np_sq[..., ::-1], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    C_clusters, C_std = _compute_color_features(rgb_for_color, mask01)
    Da, De = _compute_diameter_features(mask01)

    feat = {
        "has_segmentation": True,
        "mask_valid": True,
        "area_ratio": float(Da),
        "A_asymmetry": float(A),
        "B_border_irreg": float(B),
        "C_clusters": float(C_clusters),
        "C_color_std": float(C_std),
        "D_area_ratio": float(Da),
        "D_equiv_diam": float(De),
    }
    feat["abcd_explanation"] = _explain_abcd(feat)

    # --- Overlay: lesion original, background tinted #314026 (≈40% opacity) ---
    h0, w0 = np_sq.shape[:2]
    mask_resized = cv2.resize(mask01, (w0, h0), interpolation=cv2.INTER_NEAREST)

    overlay = np_sq.astype(np.float32)

    bg_mask = (mask_resized == 0).astype(np.float32)[..., None]  # (H,W,1)
    bg_color = np.array([0x31, 0x40, 0x26], dtype=np.float32).reshape(1, 1, 3)
    alpha = 0.4

    overlay = overlay * (1.0 - alpha * bg_mask) + bg_color * (alpha * bg_mask)

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)
    feat["mask_overlay_b64"] = _pil_to_b64(overlay_img)

    return feat


# ---------------------------------------------------------------------
# MULTI-BACKBONE Grad-CAM-like ENSEMBLE
# ---------------------------------------------------------------------
def _tensor_denorm01(x_bchw: torch.Tensor) -> np.ndarray:
    x = x_bchw[0].detach().cpu().numpy().transpose(1, 2, 0)
    x = x * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x, 0.0, 1.0)


def pick_target_layer_for_gradcam(model: nn.Module) -> nn.Module:
    m = model.backbone if hasattr(model, "backbone") else model
    name = m.__class__.__name__.lower()

    if "convnext" in name:
        if hasattr(m, "stages"):
            try:
                last_block = m.stages[-1].blocks[-1]
                for mod in reversed(list(last_block.modules())):
                    if isinstance(mod, nn.Conv2d):
                        return mod
            except Exception:
                pass
        if hasattr(m, "features"):
            for mod in reversed(list(m.features.modules())):
                if isinstance(mod, nn.Conv2d):
                    return mod

    if "efficientnet" in name and hasattr(m, "blocks"):
        try:
            last = list(m.blocks.children())[-1]
            for mod in reversed(list(last.modules())):
                if isinstance(mod, nn.Conv2d):
                    return mod
        except Exception:
            pass

    if "densenet" in name and hasattr(m, "features"):
        for cand in ["denseblock4", "denseblock3"]:
            if hasattr(m.features, cand):
                layer = getattr(m.features, cand)
                for mod in reversed(list(layer.modules())):
                    if isinstance(mod, nn.Conv2d):
                        return mod

    for container_name in ["stages", "layers", "blocks", "features"]:
        if hasattr(m, container_name):
            children = list(getattr(m, container_name).children())
            if children:
                for mod in reversed(list(children[-1].modules())):
                    if isinstance(mod, nn.Conv2d):
                        return mod

    for mod in reversed(list(m.modules())):
        if isinstance(mod, nn.Conv2d):
            return mod

    raise RuntimeError("No Conv2d layer found for Grad-CAM target.")


def _gradcam_like(
    model: nn.Module,
    image_tensor: torch.Tensor,
    class_idx: int,
    target_layer: nn.Module,
    smooth: bool = True,
) -> Optional[np.ndarray]:
    model.eval()
    feats: List[torch.Tensor] = []
    grads: List[torch.Tensor] = []

    def fwd_hook(_m, _inp, out):
        feats.append(out)

    def bwd_hook(_m, _gin, gout):
        grads.append(gout[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        x = image_tensor.clone().detach().requires_grad_(True).to(DEVICE)
        logits = model(x)
        score = logits.view(1, -1)[0, class_idx]

        model.zero_grad(set_to_none=True)
        score.backward()

        if not feats or not grads:
            return None

        activation = feats[0]
        gradient = grads[0]

        if activation.ndim == 3:
            activation = activation.unsqueeze(0)
        if gradient.ndim == 3:
            gradient = gradient.unsqueeze(0)

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1))
        cam = cam[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

        if smooth:
            cam = cv2.GaussianBlur(cam, (9, 9), 0)

        cam -= cam.min()
        if cam.max() > 0:
            cam /= (cam.max() + 1e-8)

        return cam.astype(np.float32)
    except Exception:
        return None
    finally:
        h1.remove()
        h2.remove()


def _overlay_cam_on_image(im01: np.ndarray, cam_map: np.ndarray) -> np.ndarray:
    H, W = im01.shape[:2]
    cam_resized = cv2.resize(cam_map, (W, H), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = (0.4 * heatmap + 0.6 * im01).clip(0.0, 1.0)
    return (overlay * 255).astype(np.uint8)


def multi_backbone_cam_from_bytes(
    image_bytes: bytes,
    models: Dict[str, nn.Module],
) -> Dict[str, Optional[str]]:
    if not models:
        return {"cam_b64": None}

    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_raw = np.array(pil)
    np_sq = _center_crop_square(np_raw)
    pil_sq = Image.fromarray(np_sq)

    x_cls = _cls_tfm()(pil_sq).unsqueeze(0).to(DEVICE)
    im01 = _tensor_denorm01(x_cls)

    cand_keys = ["convnext_tiny", "efficientnet_b0", "densenet121", "swin-tiny", "swin_tiny"]
    models_for_avg: List[nn.Module] = [models[k] for k in cand_keys if k in models]
    if len(models_for_avg) < 2:
        models_for_avg = list(models.values())

    cam_maps: List[np.ndarray] = []
    for m in models_for_avg:
        with torch.no_grad():
            logits = m(x_cls)
            ci = int(torch.argmax(logits, dim=1).item())
        try:
            tgt = pick_target_layer_for_gradcam(m)
            cam_map = _gradcam_like(m, x_cls, ci, tgt, smooth=True)
            if cam_map is not None:
                cam_maps.append(cam_map)
        except Exception as ex:
            print(f"[WARN] Grad-CAM failed for {m.__class__.__name__}: {ex}")

    if not cam_maps:
        return {"cam_b64": None}

    cam_avg = np.mean(cam_maps, axis=0)
    cam_avg -= cam_avg.min()
    if cam_avg.max() > 0:
        cam_avg /= (cam_avg.max() + 1e-8)

    overlay_rgb = _overlay_cam_on_image(im01, cam_avg)
    cam_img = Image.fromarray(overlay_rgb)
    return {"cam_b64": _pil_to_b64(cam_img)}
