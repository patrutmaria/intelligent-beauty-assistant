# Beauty Assistant

A web-based AI beauty consultation app that recommends makeup products and perfumes using graph neural networks (VGAE + LightGCN) trained on real Sephora product data.

---

## Overview

The app provides a personalised beauty experience: you tell it your occasion, upload a photo for skin analysis, set your preferences, and it recommends a complete makeup routine with shade-matched foundations and curated perfumes.

A **Variational Graph AutoEncoder (VGAE)** encodes relationships between 1,600 products, 200 user profiles, 30 ingredients, and 10 curated looks into a shared latent space. New users get matched through cold-start embedding similarity, with content-based boosting for skin type, undertone, finish, budget, and skin concerns.

---

## Features

- **Graph-based recommendations** -- VGAE and LightGCN trained on a heterogeneous beauty graph (products, users, ingredients, looks)
- **Skin analysis** -- upload a face photo + optional wrist/vein photo to detect undertone, skin tone depth, and skin type via MediaPipe + LAB colour analysis
- **Shade matching** -- finds the closest foundation shade from Sephora's shade database based on detected skin hex colour
- **Routine builder** -- generates a step-by-step routine (primer to perfume) in canonical application order
- **Perfume discovery** -- dedicated perfume page with scent family filtering, powered by event-based scent preference boosting
- **Concern matching** -- highlights products with active ingredients targeting selected skin concerns (acne, dark circles, redness, etc.)
- **Wishlist** -- save products across recommendations
- **Responsive UI** -- glassmorphism design with step-by-step wizard flow

---

## Project Structure

```
beauty_assistant/
├── api.py                  # FastAPI server (entry point)
├── train_vgae.py           # VGAE training script
├── requirements.txt
│
├── graph/                  # Graph neural network core
│   ├── vgae_model.py       # VGAE architecture (GCN encoder + inner-product decoder)
│   ├── graph_builder.py    # Builds PyG graph from CSV data
│   ├── trainer.py          # Training loop with edge splitting + BPR loss
│   └── recommender.py      # Cold-start inference + domain boosting
│
├── skin/                   # Skin analysis & colour matching
│   ├── face_skin_extractor.py  # MediaPipe face mesh skin tone extraction
│   ├── shade_analyzer.py       # LAB colour space undertone analysis
│   ├── shade_matcher.py        # Foundation shade matching by hex distance
│   ├── skin_classifier.py      # CNN skin type classification (optional)
│   ├── skin_heuristics.py      # Heuristic skin type + acne detection
│   └── color_advisor.py        # Colour family recs for blush/lipstick/eyeshadow
│
├── evaluation/             # Experiments & benchmarks
│   ├── eval_ranking.py     # Recall@K, NDCG@K evaluation (leave-last-out)
│   ├── eval_multiseed.py   # Multi-seed variance analysis (VGAE vs LightGCN)
│   ├── eval_experiments.py # Ablation study (KL annealing, deterministic, etc.)
│   ├── show_results.py     # Pretty-print results table
│   └── experiments_log.csv # All experiment results
│
├── scripts/                # Data generation & preprocessing
│   ├── generate_interactions.py  # Synthetic interaction generation (seed=42)
│   ├── import_sephora.py         # Sephora HuggingFace -> products.csv
│   ├── merge_shades.py           # Shade dataset merging
│   ├── fetch_sephora_images.py   # Product image fetching
│   ├── extract_product_colors.py # Dominant colour extraction via K-means
│   └── fill_missing_colors.py    # Colour gap filling
│
├── data/                   # Datasets
│   ├── products.csv        # 1,600 products (11 categories)
│   ├── interactions.csv    # 5,518 synthetic user-product interactions
│   ├── ingredients.csv     # 30 active ingredients
│   ├── looks.csv           # 10 curated makeup looks
│   └── shades.json         # Foundation shade database (~4,800 shades)
│
├── models/                 # Trained checkpoints
│   ├── vgae_beauty.pt      # VGAE model (28KB, auto-generated)
│   ├── skin_cnn/           # Skin type CNN (optional)
│   └── acne_cnn/           # Acne severity CNN (optional)
│
└── static/                 # Frontend
    ├── index.html          # Main consultation SPA
    └── perfumes.html       # Perfume discovery page
```

---

## Graph Structure

The VGAE operates on a heterogeneous graph with ~1,840 nodes and ~39,000 edges:

| Node Type   | Count | Features                           |
|-------------|-------|------------------------------------|
| Products    | 1,600 | category, price, rating, finish    |
| Users       | 200   | skin type, event preference        |
| Looks       | 10    | style, occasion                    |
| Ingredients | 30    | type, effect                       |

| Edge Type             | Source                              |
|-----------------------|-------------------------------------|
| User -- Product       | interactions.csv (like/view/purchase) |
| Product -- Look       | looks.csv product lists             |
| Product -- Ingredient | products.csv key_ingredients        |
| Product -- Product    | KNN by price within category (K=6)  |

Node features are encoded as 32-dimensional vectors combining one-hot type, category, skin type, normalised price/rating, and finish.

---

## Evaluation Results

All models evaluated with leave-last-out protocol, 3 seeds (42, 123, 256), 500 epochs.

| Method           | Recall@10     | NDCG@10       | AUC           |
|------------------|---------------|---------------|---------------|
| Random           | 0.007 ± 0.006 | 0.003 ± 0.002 | --            |
| Popularity       | 0.020         | 0.011         | --            |
| Pop+Boost        | 0.025         | 0.013         | --            |
| VGAE (BPR)       | 0.007 ± 0.002 | 0.003 ± 0.002 | 0.836 ± 0.005 |
| VGAE+KL anneal   | 0.018 ± 0.002 | 0.009 ± 0.003 | 0.906 ± 0.004 |
| **LightGCN**     | **0.103 ± 0.009** | **0.056 ± 0.004** | **0.948 ± 0.001** |

Full ablation study (4 VGAE variants) available via:
```bash
/opt/anaconda3/bin/python evaluation/show_results.py
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_vgae.py --epochs 500 --beta 0.1 --lr 0.005
```

Creates `models/vgae_beauty.pt` (~1 minute). Auto-trains on first API request if missing.

### 3. Start the server

```bash
python api.py
```

Open **http://localhost:8000**.

### 4. Run evaluation (optional)

```bash
/opt/anaconda3/bin/python evaluation/show_results.py          # view results
/opt/anaconda3/bin/python evaluation/eval_experiments.py       # re-run ablation study
/opt/anaconda3/bin/python evaluation/eval_multiseed.py         # multi-seed analysis
```

---

## Data Sources

| Source | Description |
|--------|-------------|
| **Sephora Products** (HuggingFace) | 8,494 real products, filtered to 1,600 across 11 categories |
| **The Pudding Shade Dataset** | 6,816 foundation shades with hex colours across 107 brands |
| **Synthetic interactions** | 5,518 user-product interactions generated via `scripts/generate_interactions.py` (seed=42) |

---

## Technologies

- **PyTorch + PyTorch Geometric** -- VGAE / LightGCN models
- **FastAPI + Uvicorn** -- async web server
- **MediaPipe** -- face mesh for skin tone extraction
- **Pandas / NumPy / scikit-learn** -- data processing
- **Pillow** -- image processing for shade analysis

---

## Author

**Maria Patrut** -- Master's Student in Artificial Intelligence
