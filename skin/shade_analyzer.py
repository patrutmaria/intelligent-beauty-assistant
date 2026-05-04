"""Analyzes uploaded photos to determine skin undertone and tone depth using PIL + numpy."""

import io
import numpy as np
from PIL import Image


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB array (H, W, 3) to float32 CIE-LAB. Reference white: D65."""
    r = rgb[:, :, 0] / 255.0
    g = rgb[:, :, 1] / 255.0
    b = rgb[:, :, 2] / 255.0

    def _lin(c):
        return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)

    r, g, b = _lin(r), _lin(g), _lin(b)

    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    X /= 0.9505
    Y /= 1.0000
    Z /= 1.0890

    def _f(t):
        return np.where(t > 0.008856, t ** (1.0 / 3.0), (903.3 * t + 16.0) / 116.0)

    fx, fy, fz = _f(X), _f(Y), _f(Z)

    L  = 116.0 * fy - 16.0
    a  = 500.0 * (fx - fy)
    b_ = 200.0 * (fy - fz)

    return np.stack([L, a, b_], axis=-1).astype(np.float32)


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Skin-pixel detector using YCbCr range plus brightness floor.
    The brightness floor (Y >= 90) rejects dark hair, eyebrows, and deep
    jawline shadows that would otherwise pass the chroma test.
    """
    R = rgb[:, :, 0].astype(np.float32)
    G = rgb[:, :, 1].astype(np.float32)
    B = rgb[:, :, 2].astype(np.float32)

    Y  =  0.299  * R + 0.587  * G + 0.114  * B
    Cb = -0.16874 * R - 0.33126 * G + 0.5    * B + 128
    Cr =  0.5    * R - 0.41869 * G - 0.08131 * B + 128

    return (
        (Y  >= 90)  & (Y  <= 240) &
        (Cb >= 85)  & (Cb <= 130) &
        (Cr >= 133) & (Cr <= 175) &
        (R  >= G)
    )


def _load(image_bytes: bytes, max_side: int = 512) -> np.ndarray:
    """Load image bytes -> resized uint8 RGB numpy array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return np.asarray(img)


_UNDERTONE_LABELS = {
    "warm":    "Warm Undertone",
    "cool":    "Cool Undertone",
    "neutral": "Neutral Undertone",
}

_UNDERTONE_EMOJI = {
    "warm": "🟡",
    "cool": "🔵",
    "neutral": "⚪",
}

_UNDERTONE_TIPS = {
    "warm": (
        "Your skin has golden, peachy or yellow undertones. "
        "You'll shine in warm-toned foundations with a yellow or peach base, "
        "honey/bronze highlighters, and peachy-coral blushes."
    ),
    "cool": (
        "Your skin has pink, rosy or bluish undertones. "
        "Look for foundations with a pink or neutral-cool base, "
        "rose-gold or silver highlighters, and berry or rose blushes."
    ),
    "neutral": (
        "Your skin has a balanced mix of warm and cool tones. "
        "Most foundation shades work well for you — "
        "look for 'neutral' on the label, and you can wear both warm and cool accents."
    ),
}

_LIGHTNESS_LABELS = {
    "light":        "Light skin tone",
    "light-medium": "Light-medium skin tone",
    "medium":       "Medium skin tone",
    "medium-deep":  "Medium-deep skin tone",
    "deep":         "Deep skin tone",
}


def _white_balance(rgb: np.ndarray) -> np.ndarray:
    """
    Gray-world / brightest-pixel white balance.
    Uses the brightest 5% of pixels as neutral white reference
    and applies per-channel scale factors (clamped to 0.7-1.4x).
    """
    rgb_f = rgb.astype(np.float32)
    brightness = rgb_f.mean(axis=2)
    threshold = np.percentile(brightness, 95)
    bright_mask = brightness >= threshold
    if bright_mask.sum() < 50:
        return rgb

    bright = rgb_f[bright_mask]
    mean_R = bright[:, 0].mean()
    mean_G = bright[:, 1].mean()
    mean_B = bright[:, 2].mean()
    target = (mean_R + mean_G + mean_B) / 3.0
    if target < 1:
        return rgb
    scale_R = max(0.7, min(1.4, target / max(mean_R, 1)))
    scale_G = max(0.7, min(1.4, target / max(mean_G, 1)))
    scale_B = max(0.7, min(1.4, target / max(mean_B, 1)))

    out = rgb_f.copy()
    out[:, :, 0] *= scale_R
    out[:, :, 1] *= scale_G
    out[:, :, 2] *= scale_B
    return np.clip(out, 0, 255).astype(np.uint8)


def _dominant_color(rgb_pixels: np.ndarray) -> tuple[int, int, int]:
    """
    Find the dominant lit-skin color via 3D histogram modal binning.
    Drops the darkest 60% of pixels, quantises into a 16x16x16 RGB grid,
    rejects shadow/hair bins, and picks the most populated valid bin
    weighted toward brighter values.
    """
    if len(rgb_pixels) < 20:
        return tuple(int(x) for x in np.median(rgb_pixels, axis=0))

    rgb = rgb_pixels.astype(np.int32)

    # Drop jaw shadows, hair edges, ear interior — keep brightest 40%
    brightness = rgb.mean(axis=1)
    cutoff = float(np.percentile(brightness, 60))
    bright_mask = brightness >= cutoff
    if bright_mask.sum() > 50:
        rgb = rgb[bright_mask]

    qR = np.clip(rgb[:, 0] // 16, 0, 15)
    qG = np.clip(rgb[:, 1] // 16, 0, 15)
    qB = np.clip(rgb[:, 2] // 16, 0, 15)
    bin_idx = qR * 256 + qG * 16 + qB

    counts = np.bincount(bin_idx, minlength=4096)

    bin_brightness = np.zeros(4096, dtype=np.float32)
    for r in range(16):
        for g in range(16):
            for b in range(16):
                bin_brightness[r*256 + g*16 + b] = (r + g + b) * 16 / 3.0

    valid = (bin_brightness >= 140) & (bin_brightness <= 245) & (counts > 0)
    if valid.sum() < 5:
        valid = (bin_brightness >= 110) & (bin_brightness <= 250) & (counts > 0)

    if not valid.any():
        return tuple(int(x) for x in np.median(rgb, axis=0))

    scores = np.where(valid, counts.astype(np.float32), 0.0)
    scores = scores * (bin_brightness / 200.0)
    best_bin = int(scores.argmax())

    bR = (best_bin // 256) * 16 + 8
    bG = ((best_bin // 16) % 16) * 16 + 8
    bB = (best_bin % 16) * 16 + 8

    near = (
        (np.abs(rgb[:, 0] - bR) < 20) &
        (np.abs(rgb[:, 1] - bG) < 20) &
        (np.abs(rgb[:, 2] - bB) < 20)
    )
    if near.sum() > 10:
        center = rgb[near].mean(axis=0)
        return (int(round(center[0])), int(round(center[1])), int(round(center[2])))
    return (bR, bG, bB)


def analyze_skin_photo(image_bytes: bytes) -> dict:
    """
    Analyse a face or bare-skin photo.
    Returns dict with: undertone, lightness, hex_color, mean_L, confidence, source.
    """
    rgb_raw  = _load(image_bytes)
    rgb  = _white_balance(rgb_raw)
    lab  = _rgb_to_lab(rgb)
    mask = _skin_mask(rgb)

    if mask.sum() < 200:
        h, w = rgb.shape[:2]
        cy, cx = h // 2, w // 2
        r1, r2 = max(0, cy - h // 4), min(h, cy + h // 4)
        c1, c2 = max(0, cx - w // 4), min(w, cx + w // 4)
        skin_lab = lab[r1:r2, c1:c2].reshape(-1, 3)
        skin_rgb = rgb[r1:r2, c1:c2].reshape(-1, 3)
        confidence = 0.45
    else:
        skin_lab   = lab[mask]
        skin_rgb   = rgb[mask]
        confidence = float(min(1.0, mask.sum() / max(rgb.shape[0] * rgb.shape[1], 1) * 4))

    # Use only the brightest 25% of skin pixels (lit forehead/cheekbones),
    # excluding shadow band that would skew the median
    skin_brightness = skin_lab[:, 0]
    if len(skin_brightness) > 100:
        cutoff = np.percentile(skin_brightness, 75)
        sel    = skin_brightness >= cutoff
        skin_lab = skin_lab[sel]
        skin_rgb = skin_rgb[sel]

    mean_L = float(skin_lab[:, 0].mean())
    mean_a = float(skin_lab[:, 1].mean())
    mean_b = float(skin_lab[:, 2].mean())

    # Undertone via b*/a* ratio with adaptive thresholds per skin depth.
    # Darker skin tones have naturally different b*/a* distributions,
    # so fixed thresholds misclassify deep skin as warm.
    if mean_a > 0:
        ratio = mean_b / mean_a
    else:
        ratio = 1.85

    if mean_L < 40:
        lightness = "deep"
        warm_thresh, cool_thresh = 2.10, 1.80  
    elif mean_L < 52:
        lightness = "medium-deep"
        warm_thresh, cool_thresh = 2.05, 1.75
    elif mean_L < 62:
        lightness = "medium"
        warm_thresh, cool_thresh = 2.00, 1.70
    elif mean_L < 72:
        lightness = "light-medium"
        warm_thresh, cool_thresh = 1.95, 1.65
    else:
        lightness = "light"
        warm_thresh, cool_thresh = 1.90, 1.60

    if ratio >= warm_thresh:
        undertone = "warm"
    elif ratio <= cool_thresh:
        undertone = "cool"
    else:
        undertone = "neutral"

    mid = (warm_thresh + cool_thresh) / 2
    spread = (warm_thresh - cool_thresh) / 2
    dist_from_mid = abs(ratio - mid)
    undertone_confidence = min(0.95, 0.55 + (dist_from_mid / spread) * 0.30)

    avg_rgb = _dominant_color(skin_rgb)
    hex_color = "#{:02x}{:02x}{:02x}".format(int(avg_rgb[0]), int(avg_rgb[1]), int(avg_rgb[2]))

    return {
        "undertone":  undertone,
        "lightness":  lightness,
        "hex_color":  hex_color,
        "mean_L":     round(mean_L, 1),
        "mean_a":     round(mean_a, 2),
        "mean_b":     round(mean_b, 2),
        "confidence": round(max(confidence, undertone_confidence), 2),
        "undertone_confidence": round(undertone_confidence, 2),
        "source":     "skin",
    }


def analyze_vein_photo(image_bytes: bytes) -> dict:
    """
    Analyse a wrist/vein photo to determine undertone via the Benefit Cosmetics
    vein-color method. Compares vein pixels' B-G channel difference relative to
    the skin baseline to classify as warm/cool/neutral.
    Returns dict with: undertone, confidence, blue_score, green_score, source.
    """
    import numpy as np

    rgb = _load(image_bytes)

    skin_mask = _skin_mask(rgb)
    if skin_mask.sum() < 200:
        return {
            "undertone": "neutral", "confidence": 0.25,
            "blue_score": 0.0, "green_score": 0.0,
            "source": "vein", "error": "too few skin pixels"
        }

    skin_pixels = rgb[skin_mask].astype(np.float32)
    R, G, B = skin_pixels[:, 0], skin_pixels[:, 1], skin_pixels[:, 2]

    base_R, base_G, base_B = float(np.median(R)), float(np.median(G)), float(np.median(B))

    # Vein candidates: darker pixels with shifted B/G balance
    L = (R + G + B) / 3
    base_L = (base_R + base_G + base_B) / 3
    darker = L < base_L - 4
    rel_BG = (B - G) - (base_B - base_G)

    vein_mask = darker
    if vein_mask.sum() < 50:
        vein_mask = L <= np.percentile(L, 15)

    if vein_mask.sum() < 30:
        return {
            "undertone": "neutral", "confidence": 0.30,
            "blue_score": 0.0, "green_score": 0.0,
            "source": "vein"
        }

    band_rel = rel_BG[vein_mask]
    mean_shift = float(band_rel.mean())

    vR = float(R[vein_mask].mean())
    vG = float(G[vein_mask].mean())
    vB = float(B[vein_mask].mean())

    # Multi-signal scoring for more robust classification
    signals = []

    # Signal 1: B-G shift (primary)
    if mean_shift > 0.5:
        signals.append(("cool", min(0.95, mean_shift / 4.0 + 0.55)))
    elif mean_shift < -0.5:
        signals.append(("warm", min(0.95, abs(mean_shift) / 4.0 + 0.55)))
    else:
        signals.append(("neutral", 0.55))

    # Signal 2: Histogram peak analysis — where does the B-G distribution peak?
    hist_bg = band_rel[np.isfinite(band_rel)]
    if len(hist_bg) > 50:
        p25, p75 = np.percentile(hist_bg, 25), np.percentile(hist_bg, 75)
        # If 75th percentile is strongly blue, even median being neutral → cool
        if p75 > 1.5:
            signals.append(("cool", 0.65))
        elif p25 < -1.5:
            signals.append(("warm", 0.65))

    # Signal 3: Vein color in LAB space — blue veins have negative b*
    vein_lab_b = 0.0
    if vB > 0:
        # Quick LAB b* estimate for vein color
        vein_lab_b = (vB / 255.0) - (vG / 255.0)
        if vein_lab_b > 0.05:
            signals.append(("cool", 0.60))
        elif vein_lab_b < -0.05:
            signals.append(("warm", 0.60))

    # Combine signals with weighted voting
    vote = {"warm": 0.0, "cool": 0.0, "neutral": 0.0}
    for ut, conf in signals:
        vote[ut] += conf
    winner = max(vote, key=vote.get)
    total_vote = sum(vote.values())
    confidence = round(min(0.95, vote[winner] / max(total_vote, 0.01)), 2)

    return {
        "undertone":   winner,
        "confidence":  confidence,
        "blue_score":  round(mean_shift, 2),
        "green_score": round(-mean_shift, 2),
        "vein_rgb":    [int(round(vR)), int(round(vG)), int(round(vB))],
        "skin_baseline_rgb": [int(round(base_R)), int(round(base_G)), int(round(base_B))],
        "n_signals":   len(signals),
        "source":      "vein",
    }


def combine_results(skin: dict | None, vein: dict | None) -> dict:
    """
    Merge skin-photo and vein-photo results into a final verdict.
    Vein signal is weighted higher (0.60 vs 0.40) as it uses the dedicated
    Benefit-method colour shift detector.
    """
    if skin is None and vein is None:
        return {
            "undertone": "neutral", "lightness": "medium",
            "hex_color": "#c8956b", "confidence": 0.0,
            "tip": _UNDERTONE_TIPS["neutral"],
            "label": _UNDERTONE_LABELS["neutral"],
            "emoji": _UNDERTONE_EMOJI["neutral"],
        }

    if vein is None:
        result = dict(skin)
    elif skin is None:
        result = {**vein, "lightness": "medium", "hex_color": "#c8956b"}
    else:
        votes = {"warm": 0.0, "cool": 0.0, "neutral": 0.0}
        votes[skin["undertone"]] += skin["confidence"] * 0.40
        votes[vein["undertone"]] += vein["confidence"] * 0.60

        final = max(votes, key=votes.get)
        conf  = round(votes[final], 2)
        result = {
            **skin,
            "undertone":  final,
            "confidence": conf,
            "source":     "combined",
        }
        result["skin_undertone"] = skin["undertone"]
        result["vein_undertone"] = vein["undertone"]
        result["vein_confidence"] = vein.get("confidence", 0)

    undertone = result["undertone"]
    result["label"]     = _UNDERTONE_LABELS[undertone]
    result["emoji"]     = _UNDERTONE_EMOJI[undertone]
    result["tip"]       = _UNDERTONE_TIPS[undertone]
    result["lightness_label"] = _LIGHTNESS_LABELS.get(result.get("lightness", "medium"), "")

    return result
