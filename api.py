"""
Beauty Assistant - FastAPI server
Start: python api.py -> http://localhost:8000
"""

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from typing import Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
_STATIC = os.path.join(_ROOT, "static")

#VGAE recommender (lazy-loaded)

_recommender = None
_vgae_ready = False


def _load_recommender():
    global _recommender, _vgae_ready
    try:
        import torch, torch_geometric  # noqa
    except ImportError:
        print("[API] torch/torch_geometric not installed - VGAE unavailable.")
        return

    model_path = os.path.join(_ROOT, "models", "vgae_beauty.pt")
    data_dir = os.path.join(_ROOT, "data") + os.sep

    if not os.path.exists(model_path):
        print("[API] No trained model - training now (~1 min)...")
        from graph.graph_builder import BeautyGraphBuilder
        from graph.trainer import train_vgae
        builder = BeautyGraphBuilder(data_dir)
        graph_data, _, _ = builder.build()
        train_vgae(graph_data, model_save_path=model_path)

    from graph.recommender import BeautyRecommender
    _recommender = BeautyRecommender(model_path=model_path, data_dir=data_dir)
    _vgae_ready = True
    print("[API] VGAE recommender loaded.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    await asyncio.get_event_loop().run_in_executor(None, _load_recommender)
    yield


app = FastAPI(title="Beauty Assistant", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=_STATIC), name="static")

#Helpers

_EVENT_NORM = {
    "wedding": "wedding", "bridal": "wedding",
    "office": "office", "work": "office",
    "evening": "evening", "night": "evening", "party": "evening",
    "natural": "natural", "everyday": "natural", "casual": "natural",
    "glam": "glam", "date": "glam",
    "festival": "festival",
}

_SKIN_KEYWORDS = {
    "oily":        ["shiny", "greasy", "breakout", "acne", "large pore", "oily"],
    "dry":         ["flaky", "tight", "rough", "dull", "itchy", "dry", "dehydrated"],
    "combination": ["t-zone", "combo", "combination", "both dry and oily", "mixed"],
}


def _detect_skin(desc: str) -> str:
    d = desc.lower()
    for skin_type, kws in _SKIN_KEYWORDS.items():
        if any(k in d for k in kws):
            return skin_type
    return "normal"


def _json(obj) -> JSONResponse:
    return JSONResponse(content=json.loads(
        json.dumps(obj, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    ))


_FALLBACK = {
    "wedding": {"oily": "Estee Lauder Double Wear", "dry": "Dior Forever Skin Glow",
                "combination": "MAC Studio Fix", "normal": "NARS Sheer Glow"},
    "office":  {"oily": "L'Oreal Infallible Matte", "dry": "Maybelline Dream Satin",
                "combination": "Revlon ColorStay", "normal": "NYX Born to Glow"},
    "natural": {"oily": "Neutrogena Skin Tint", "dry": "Garnier BB Cream",
                "combination": "L'Oreal True Match", "normal": "Clinique Even Better"},
    "evening": {"oily": "Fenty Soft Matte", "dry": "Charlotte Tilbury Light Wonder",
                "combination": "Milani Conceal+Perfect", "normal": "Too Faced Born This Way"},
    "glam":    {"oily": "Fenty Soft Matte", "dry": "Charlotte Tilbury Airbrush",
                "combination": "MAC Studio Fix", "normal": "NARS Natural Radiant"},
    "festival":{"oily": "Urban Decay All Nighter", "dry": "IT Cosmetics CC+ Cream",
                "combination": "Smashbox Studio Skin", "normal": "Clinique Even Better"},
}


def _fallback_recommend(skin_type: str, event_type: str) -> dict:
    ev = _EVENT_NORM.get(event_type.lower(), "natural")
    sk = skin_type if skin_type in ("oily", "dry", "combination", "normal") else "normal"
    name = _FALLBACK.get(ev, _FALLBACK["natural"]).get(sk, "NARS Natural Radiant Foundation")
    return {
        "user_profile": {"skin_type": skin_type, "event_type": event_type, "budget": None},
        "recommended_look": {"look_id": 0, "name": f"{ev.capitalize()} Look",
                             "style": ev, "occasion": ev, "score": 0.8},
        "alternative_looks": [],
        "recommended_products": [
            {"product_id": 0, "name": name, "brand": "-", "category": "foundation",
             "price": 0.0, "finish": "-", "rating": 4.5, "score": 0.8,
             "reason": f"classic pick for {sk} skin"}
        ],
        "detected_skin_type": skin_type, "_fallback": True,
    }


#Routes

@app.get("/favicon.ico")
async def favicon():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><text y="26" font-size="28">\u2743</text></svg>'
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(_STATIC, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        })


@app.get("/perfumes", response_class=HTMLResponse)
async def perfumes_page():
    with open(os.path.join(_STATIC, "perfumes.html"), encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        })


#Recommend

class RecommendRequest(BaseModel):
    skin_description: str
    event_type: str
    budget: Optional[float] = None
    undertone: Optional[str] = None
    lightness: Optional[str] = None
    categories: Optional[list[str]] = None
    top_k_per_category: Optional[int] = 1
    concerns: Optional[list[str]] = None
    acne: Optional[str] = None
    cnn_skin_type: Optional[str] = None
    skin_hex: Optional[str] = None


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    skin_type = req.cnn_skin_type or _detect_skin(req.skin_description)
    event_norm = _EVENT_NORM.get(req.event_type.lower(), req.event_type)

    if _vgae_ready and _recommender is not None:
        try:
            result = _recommender.recommend_for_new_user(
                skin_type=skin_type, event_type=event_norm, budget=req.budget,
                undertone=req.undertone, lightness=req.lightness,
                concerns=req.concerns, acne=req.acne, skin_hex=req.skin_hex,
                top_k_products=5, top_k_looks=3,
            )
            result["detected_skin_type"] = skin_type
            return _json(result)
        except Exception as e:
            print(f"[API] VGAE error: {e} - falling back to rule-based")

    return _json(_fallback_recommend(skin_type, event_norm))


#Routine

@app.post("/routine")
async def routine(req: RecommendRequest):
    skin_type = req.cnn_skin_type or _detect_skin(req.skin_description)
    event_norm = _EVENT_NORM.get(req.event_type.lower(), req.event_type)

    if not (_vgae_ready and _recommender is not None):
        return JSONResponse(content={"error": "VGAE model not loaded", "step_count": 0, "routine_steps": []}, status_code=503)

    try:
        result = _recommender.build_routine(
            skin_type=skin_type, event_type=event_norm, budget=req.budget,
            undertone=req.undertone, lightness=req.lightness,
            concerns=req.concerns, acne=req.acne, skin_hex=req.skin_hex,
            categories=req.categories, top_k_per_category=req.top_k_per_category or 1,
        )
        result["detected_skin_type"] = skin_type
        return _json(result)
    except Exception as e:
        print(f"[API] Routine error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


#Undertone analysis

@app.post("/analyze-undertone")
async def analyze_undertone(
    skin_photo: Optional[UploadFile] = File(default=None),
    vein_photo: Optional[UploadFile] = File(default=None),
):
    if skin_photo is None and vein_photo is None:
        raise HTTPException(status_code=422, detail="Provide at least one photo.")

    from skin.shade_analyzer import analyze_skin_photo, analyze_vein_photo, combine_results
    from skin.face_skin_extractor import extract_skin_tone
    from skin.skin_heuristics import analyze as heuristic_analyze
    try:
        from skin.skin_classifier import classify as cnn_classify
    except Exception:
        cnn_classify = None

    skin_result = vein_result = None
    skin_bytes = None

    try:
        if skin_photo is not None:
            skin_bytes = await skin_photo.read()
            if skin_bytes:
                skin_result = analyze_skin_photo(skin_bytes)
    except Exception as e:
        print(f"[API] Skin photo analysis failed: {e}")

    try:
        if vein_photo is not None:
            data = await vein_photo.read()
            if data:
                vein_result = analyze_vein_photo(data)
    except Exception as e:
        print(f"[API] Vein photo analysis failed: {e}")

    final = combine_results(skin_result, vein_result)

    if skin_bytes:
        # MediaPipe face skin tone extraction
        skin_hex_value = None
        try:
            face = extract_skin_tone(skin_bytes)
            if face.get("available"):
                skin_hex_value = face["hex"]
                final["skin_rgb"] = face["rgb"]
                final["skin_regions"] = face.get("regions")
                final["skin_source"] = "mediapipe"
        except Exception as e:
            print(f"[API] MediaPipe extraction failed: {e}")

        if not skin_hex_value:
            skin_hex_value = final.get("hex_color")
            final["skin_source"] = "lab"

        if skin_hex_value:
            final["skin_hex"] = skin_hex_value
            final["hex_color"] = skin_hex_value

        # Heuristic skin type + acne detection with optional CNN cross-check
        try:
            h = heuristic_analyze(skin_bytes)
            if h.get("available"):
                heur_skin = h["skin_type"]
                heur_acne = h["acne"]
                heur_skin_conf = h["skin_confidence"]
                heur_acne_conf = h["acne_confidence"]

                cnn_skin = cnn_acne = None
                if cnn_classify is not None:
                    try:
                        cnn = cnn_classify(skin_bytes)
                        if cnn.get("available"):
                            cnn_skin = cnn.get("skin_type")
                            cnn_acne = cnn.get("acne")
                    except Exception:
                        pass

                # Boost confidence when CNN agrees
                if cnn_skin == heur_skin and heur_skin is not None:
                    heur_skin_conf = round(min(0.95, heur_skin_conf + 0.10), 2)
                if cnn_acne == heur_acne and heur_acne is not None:
                    heur_acne_conf = round(min(0.95, heur_acne_conf + 0.10), 2)

                final["suggested_skin_type"] = heur_skin
                final["suggested_skin_conf"] = heur_skin_conf
                final["suggested_acne"] = heur_acne
                final["suggested_acne_conf"] = heur_acne_conf
                final["suggested_signals"] = h.get("skin_signals", {})
                if cnn_skin and cnn_skin != heur_skin:
                    final["cnn_disagreement"] = {"cnn_skin_type": cnn_skin}
        except Exception as e:
            print(f"[API] Skin heuristic failed: {e}")

    return JSONResponse(content=final)


#Perfume recommendations

class PerfumeRequest(BaseModel):
    event_type: str
    budget: Optional[float] = None
    scent_families: Optional[list[str]] = None
    unisex_only: Optional[bool] = False
    top_k: Optional[int] = 6


@app.post("/recommend-perfumes")
async def recommend_perfumes(req: PerfumeRequest):
    event_norm = _EVENT_NORM.get(req.event_type.lower(), req.event_type)

    if not (_vgae_ready and _recommender is not None):
        return JSONResponse(content={"error": "VGAE model not loaded", "perfumes": [], "count": 0}, status_code=503)

    try:
        result = _recommender.build_routine(
            skin_type="normal", event_type=event_norm, budget=req.budget,
            categories=["perfume"], top_k_per_category=req.top_k or 8,
            preferred_scents=req.scent_families,
        )
        perfumes = result.get("routine_steps", [])
        if req.unisex_only:
            perfumes = [p for p in perfumes if p.get("is_unisex")]
        perfumes = perfumes[:req.top_k or 6]

        return _json({"event_type": event_norm, "perfumes": perfumes, "count": len(perfumes)})
    except Exception as e:
        print(f"[API] Perfume recommendation error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _vgae_ready}


if __name__ == "__main__":
    import uvicorn
    print("\n  Beauty Assistant API")
    print("  http://localhost:8000\n")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
