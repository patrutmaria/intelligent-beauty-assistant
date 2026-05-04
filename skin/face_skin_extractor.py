"""Extracts skin tone from face photos using MediaPipe face mesh + YCbCr filtering."""

import io
import os

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Outer ring of the face contour (excludes hair).
# Source: https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py

FACE_OVAL_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

_face_mesh = None

def _get_mesh():
    global _face_mesh
    if _face_mesh is None:
        import mediapipe as mp
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.4,
        )
    return _face_mesh


def extract_skin_tone(image_bytes: bytes) -> dict:
    """
    Find face contour with MediaPipe, build a polygon mask of the face oval,
    refine to actual skin pixels with YCbCr, and return the dominant lit-skin
    color via histogram modal binning.
    """
    try:
        import numpy as np
        import cv2
        from PIL import Image
    except ImportError as e:
        return {"available": False, "error": f"missing dep: {e}"}

    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.asarray(pil)
        h, w = rgb.shape[:2]
    except Exception as e:
        return {"available": False, "error": f"decode: {e}"}

    try:
        mesh = _get_mesh()
        result = mesh.process(rgb)
        if not result.multi_face_landmarks:
            return {"available": False, "error": "no face detected — using LAB hex instead"}

        landmarks = result.multi_face_landmarks[0].landmark

        polygon = np.array(
            [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in FACE_OVAL_LANDMARKS],
            dtype=np.int32,
        )
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(face_mask, [polygon], 255)

        # Shrink inward to avoid hairline edge
        kernel = np.ones((9, 9), np.uint8)
        face_mask = cv2.erode(face_mask, kernel, iterations=2)

        # Exclude eyes, brows, and lips
        EYES_LEFT  = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        EYES_RIGHT = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
        LIPS       = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        BROW_LEFT  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        BROW_RIGHT = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]

        for feature in (EYES_LEFT, EYES_RIGHT, LIPS, BROW_LEFT, BROW_RIGHT):
            poly = np.array(
                [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in feature if i < len(landmarks)],
                dtype=np.int32,
            )
            if len(poly) >= 3:
                cv2.fillPoly(face_mask, [poly], 0)
                feature_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(feature_mask, [poly], 255)
                feature_mask = cv2.dilate(feature_mask, np.ones((7, 7), np.uint8), iterations=2)
                face_mask[feature_mask > 0] = 0

        # Apply YCbCr skin filter to drop remaining non-skin pixels
        R = rgb[:, :, 0].astype(np.float32)
        G = rgb[:, :, 1].astype(np.float32)
        B = rgb[:, :, 2].astype(np.float32)
        Y  =  0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.16874 * R - 0.33126 * G + 0.5    * B + 128
        Cr =  0.5    * R - 0.41869 * G - 0.08131 * B + 128
        ycbcr_skin = (
            (Y >= 90) & (Y <= 240) &
            (Cb >= 85) & (Cb <= 130) &
            (Cr >= 133) & (Cr <= 175) &
            (R >= G)
        )

        combined_mask = (face_mask > 0) & ycbcr_skin
        skin_pixels = rgb[combined_mask]

        if len(skin_pixels) < 200:
            return {"available": False, "error": "not enough lit skin pixels in face oval"}

        from .shade_analyzer import _dominant_color

        REGIONS = {
            "forehead":    [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 148, 152],
            "left_cheek":  [187, 207, 206, 205, 50, 117, 118, 119, 100, 142, 203, 36, 101],
            "right_cheek": [411, 427, 426, 425, 280, 346, 347, 348, 329, 371, 423, 266, 330],
            "chin":        [152, 148, 176, 149, 150, 136, 172, 58, 377, 400, 378, 379, 365, 397],
        }

        region_colors = {}
        for region_name, region_landmarks in REGIONS.items():
            valid_lms = [i for i in region_landmarks if i < len(landmarks)]
            if len(valid_lms) < 3:
                continue
            region_poly = np.array(
                [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in valid_lms],
                dtype=np.int32)
            region_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(region_mask, [region_poly], 255)
            region_mask = cv2.erode(region_mask, np.ones((5, 5), np.uint8), iterations=1)
            region_skin = (region_mask > 0) & ycbcr_skin
            pixels = rgb[region_skin]
            if len(pixels) >= 30:
                rc, gc, bc = _dominant_color(pixels)
                region_colors[region_name] = (int(rc), int(gc), int(bc))

        if region_colors:
            weights = {"forehead": 0.15, "left_cheek": 0.30, "right_cheek": 0.30, "chin": 0.25}
            total_w = sum(weights.get(k, 0.25) for k in region_colors)
            avg_r = sum(region_colors[k][0] * weights.get(k, 0.25) for k in region_colors) / total_w
            avg_g = sum(region_colors[k][1] * weights.get(k, 0.25) for k in region_colors) / total_w
            avg_b = sum(region_colors[k][2] * weights.get(k, 0.25) for k in region_colors) / total_w
            r, g, b = int(round(avg_r)), int(round(avg_g)), int(round(avg_b))
        else:
            r, g, b = _dominant_color(skin_pixels)
            r, g, b = int(r), int(g), int(b)

        return {
            "available": True,
            "rgb":       (r, g, b),
            "hex":       _rgb_to_hex((r, g, b)),
            "n_pixels":  int(combined_mask.sum()),
            "regions":   {k: _rgb_to_hex(v) for k, v in region_colors.items()},
        }
    except Exception as e:
        return {"available": False, "error": f"mediapipe: {e}"}


def _ycbcr_fallback(rgb, reason: str) -> dict:
    """
    No face detected -> restrict search to center 60% of image and sample
    skin-coloured pixels there. Uses median brightness for shadow robustness.
    """
    import numpy as np

    h, w = rgb.shape[:2]
    cy, cx = h // 2, w // 2
    half_h, half_w = int(h * 0.30), int(w * 0.30)
    r1 = max(0, cy - half_h); r2 = min(h, cy + half_h)
    c1 = max(0, cx - half_w); c2 = min(w, cx + half_w)
    center = rgb[r1:r2, c1:c2]

    R = center[:, :, 0].astype(np.float32)
    G = center[:, :, 1].astype(np.float32)
    B = center[:, :, 2].astype(np.float32)
    Y  =  0.299  * R + 0.587  * G + 0.114  * B
    Cb = -0.16874 * R - 0.33126 * G + 0.5     * B + 128
    Cr =  0.5    * R - 0.41869 * G - 0.08131 * B + 128
    mask = (
        (Y  >= 40)  & (Y  <= 240) &
        (Cb >= 80)  & (Cb <= 130) &
        (Cr >= 130) & (Cr <= 180)
    )
    skin = center[mask]

    if len(skin) < 100:
        R = rgb[:, :, 0].astype(np.float32)
        G = rgb[:, :, 1].astype(np.float32)
        B = rgb[:, :, 2].astype(np.float32)
        Y  =  0.299  * R + 0.587  * G + 0.114  * B
        Cb = -0.16874 * R - 0.33126 * G + 0.5     * B + 128
        Cr =  0.5    * R - 0.41869 * G - 0.08131 * B + 128
        mask = (
            (Y  >= 40)  & (Y  <= 240) &
            (Cb >= 80)  & (Cb <= 130) &
            (Cr >= 130) & (Cr <= 180)
        )
        skin = rgb[mask]

    if len(skin) < 50:
        return {"available": False, "error": f"{reason}; no skin pixels"}

    # Use brighter half of skin pixels for robustness against shadows
    brightness = skin.mean(axis=1)
    bright_threshold = np.percentile(brightness, 50)
    bright_mask = brightness >= bright_threshold
    if bright_mask.sum() > 30:
        skin = skin[bright_mask]

    r = int(round(np.median(skin[:, 0])))
    g = int(round(np.median(skin[:, 1])))
    b = int(round(np.median(skin[:, 2])))
    return {
        "available": True,
        "rgb":       (r, g, b),
        "hex":       _rgb_to_hex((r, g, b)),
        "n_pixels":  int(skin.shape[0]),
        "regions":   None,
        "fallback":  reason,
    }


def _rgb_to_hex(rgb) -> str:
    r, g, b = (int(round(c)) for c in rgb)
    r = max(0, min(255, r)); g = max(0, min(255, g)); b = max(0, min(255, b))
    return f"#{r:02x}{g:02x}{b:02x}"
