"""Rule-based skin type and acne detector using pixel-level signals."""

import io
import os
import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")


def analyze(image_bytes: bytes) -> dict:
    """
    Returns skin type (oily/dry/combination/normal) and acne severity
    (low/moderate/severe) with confidence scores, or an error dict.
    """
    try:
        from PIL import Image
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.asarray(pil)
    except Exception as e:
        return {"available": False, "error": f"decode: {e}"}

    h, w = rgb.shape[:2]

    # Crop to centre 70% to focus on face
    cy, cx = h // 2, w // 2
    half_h = int(h * 0.35); half_w = int(w * 0.35)
    r1 = max(0, cy - half_h); r2 = min(h, cy + half_h)
    c1 = max(0, cx - half_w); c2 = min(w, cx + half_w)
    face = rgb[r1:r2, c1:c2]

    skin_mask = _skin_mask_ycbcr(face)
    if skin_mask.sum() < 200:
        return {"available": False, "error": "no skin region found"}

    # Signal 1: glare score (oily indicator) — percentage of very bright skin pixels
    brightness = face.mean(axis=2)
    skin_brightness = brightness[skin_mask]
    p90 = np.percentile(skin_brightness, 90)
    very_bright = skin_brightness >= p90 * 1.05
    glare_score = float(very_bright.sum() / max(len(skin_brightness), 1))

    brightness_std = float(skin_brightness.std())

    # Signal 2: texture/edge score (dry indicator) — approximate Sobel edge magnitude
    bf = brightness.astype(np.float32)
    gx = np.abs(np.diff(bf, axis=1, prepend=bf[:, :1]))
    gy = np.abs(np.diff(bf, axis=0, prepend=bf[:1, :]))
    edge_mag = (gx + gy)
    skin_edges = edge_mag[skin_mask]
    texture_score = float(np.percentile(skin_edges, 75))

    # Signal 3: saturation (red dominance / dehydration cue)
    R = face[:, :, 0].astype(np.float32)
    G = face[:, :, 1].astype(np.float32)
    B = face[:, :, 2].astype(np.float32)
    sat = (np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)) / (np.maximum(np.maximum(R, G), B) + 1)
    skin_sat = sat[skin_mask]
    sat_mean = float(skin_sat.mean())

    # Specular gap: difference between p99 and p50 brightness — measures
    # sebum-induced specular highlights rather than anatomical variation
    p50 = float(np.percentile(skin_brightness, 50))
    p99 = float(np.percentile(skin_brightness, 99))
    specular_gap = p99 - p50

    is_oily = specular_gap >= 35 and glare_score >= 0.04
    is_dry  = texture_score >= 14.0 and sat_mean < 0.25 and not is_oily

    if is_oily and texture_score >= 12.0:
        skin_type  = "combination"
        skin_conf  = 0.65
    elif is_oily:
        skin_type  = "oily"
        skin_conf  = float(min(0.85, 0.55 + (specular_gap - 35) / 40))
    elif is_dry:
        skin_type  = "dry"
        skin_conf  = float(min(0.85, 0.55 + (texture_score - 14) / 30))
    elif specular_gap >= 25 and glare_score >= 0.02:
        skin_type  = "combination"
        skin_conf  = 0.6
    else:
        skin_type  = "normal"
        skin_conf  = 0.65

    # Acne detection: pixels significantly redder than median skin
    skin_R = R[skin_mask]
    skin_G = G[skin_mask]
    skin_B = B[skin_mask]

    # Redness ratio baseline is ~1.3-1.5 for normal skin, 1.6+ for blemishes
    redness_ratio = skin_R / ((skin_G + skin_B) / 2 + 1)
    median_ratio  = float(np.median(redness_ratio))
    is_blemish = redness_ratio > (median_ratio + 0.15)
    n_red = int(is_blemish.sum())
    n_skin = int(skin_mask.sum())
    red_pct = n_red / max(n_skin, 1)

    # Thresholds: ~3-5% noise, 5-10% mild, 10-18% moderate, >18% severe
    if red_pct >= 0.18:
        acne = "severe"
        acne_conf = float(min(0.85, 0.55 + red_pct))
    elif red_pct >= 0.10:
        acne = "moderate"
        acne_conf = 0.70
    else:
        acne = "low"
        acne_conf = float(min(0.85, 0.65 + (0.10 - red_pct)))

    return {
        "available":       True,
        "skin_type":       skin_type,
        "skin_confidence": round(skin_conf, 2),
        "skin_signals": {
            "glare_score":    round(glare_score, 3),
            "brightness_std": round(brightness_std, 1),
            "specular_gap":   round(specular_gap, 1),
            "texture_score":  round(texture_score, 1),
            "saturation":     round(sat_mean, 3),
        },
        "acne":            acne,
        "acne_confidence": round(acne_conf, 2),
        "acne_signals": {
            "red_pixel_pct": round(red_pct * 100, 2),
        },
    }


def _skin_mask_ycbcr(rgb: np.ndarray) -> np.ndarray:
    R = rgb[:, :, 0].astype(np.float32)
    G = rgb[:, :, 1].astype(np.float32)
    B = rgb[:, :, 2].astype(np.float32)
    Y  =  0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.16874 * R - 0.33126 * G + 0.5 * B + 128
    Cr =  0.5    * R - 0.41869 * G - 0.08131 * B + 128
    return ((Y >= 40) & (Y <= 240) & (Cb >= 80) & (Cb <= 130) & (Cr >= 130) & (Cr <= 180))
