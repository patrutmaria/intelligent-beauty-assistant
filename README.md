# Beauty Assistant

A web-based AI beauty consultation app that recommends makeup products and perfumes using a **Variational Graph AutoEncoder (VGAE)** trained on real Sephora product data.

---

## Overview

The app provides a personalised beauty experience: you tell it your occasion, upload a photo for skin analysis, set your preferences, and it recommends a complete makeup routine with shade-matched foundations and curated perfumes.

Under the hood, a **graph neural network** encodes relationships between 1,600 products, 5,500 user interactions, 30 ingredients, and 10 curated looks into a shared latent space. New users get matched through cold-start embedding similarity, with content-based boosting for skin type, undertone, finish, budget, and skin concerns.

---

## Features

- **VGAE Recommendations** -- graph-based product scoring trained on a heterogeneous beauty graph (products, users, ingredients, looks)
- **Skin Analysis** -- upload a face photo + optional wrist/vein photo to detect undertone, skin tone depth, and skin type using MediaPipe + LAB colour analysis
- **Shade Matching** -- finds the closest foundation shade from Sephora's shade database based on your detected skin hex colour
- **Complete Routine Builder** -- generates a step-by-step routine (primer to perfume) in canonical application order
- **Perfume Discovery** -- dedicated perfume page with scent family filtering (floral, woody, warm & spicy, citrus, gourmand, aromatic), powered by the same VGAE with event-based scent preference boosting
- **Concern Matching** -- highlights products with active ingredients that target your selected skin concerns (acne, dark circles, redness, etc.)
- **Wishlist** -- save products across recommendations
- **Responsive UI** -- glassmorphism design with Cormorant Garamond typography, step-by-step wizard flow

---

## Architecture

```
Browser (index.html / perfumes.html)
  |
  v
FastAPI Server (api.py, port 8000)
  |
  +-- GET  /              --> main consultation page
  +-- GET  /perfumes      --> perfume discovery page
  +-- POST /recommend     --> VGAE product recommendations
  +-- POST /routine       --> step-by-step routine builder
  +-- POST /recommend-perfumes --> perfume recommendations with scent filtering
  +-- POST /analyze-undertone  --> photo-based skin analysis
  +-- GET  /health        --> model status
  |
  v
VGAE Engine (graph/)
  |
  +-- graph_builder.py    --> builds heterogeneous graph from CSV data
  +-- vgae_model.py       --> 2-layer GCN encoder, inner-product decoder
  +-- trainer.py          --> training loop with edge splitting + early stopping
  +-- recommender.py      --> cold-start inference, content boosting, routine builder
  +-- shade_matcher.py    --> foundation shade matching by hex distance
  +-- shade_analyzer.py   --> LAB colour space undertone analysis
  +-- face_skin_extractor.py --> MediaPipe face mesh skin tone extraction
  +-- skin_heuristics.py  --> heuristic skin type + acne detection
  +-- color_advisor.py    --> colour family recommendations for blush/lipstick/eyeshadow
```

---

## Graph Structure

The VGAE operates on a heterogeneous graph with ~7,100 nodes and ~50,000 edges:

| Node Type   | Count  | Features                                          |
|-------------|--------|---------------------------------------------------|
| Users       | ~5,500 | skin type, event preference                       |
| Products    | 1,600  | category, price, rating, finish, skin type        |
| Looks       | 10     | style, occasion                                   |
| Ingredients | 30     | type (humectant, emollient, etc.)                 |

| Edge Type             | Source                              |
|-----------------------|-------------------------------------|
| User -- Product       | interactions.csv (like/view/purchase) |
| Product -- Look       | looks.csv product lists             |
| Product -- Ingredient | products.csv key_ingredients        |
| Product -- Product    | KNN by price within category (K=6)  |

Node features are encoded as 32-dimensional vectors combining one-hot type, category, skin type, normalised price/rating, and finish embeddings.

---

## Data Sources & Processing

The app's data comes from **three real-world datasets** plus curated supplements. All raw data was processed through scripts in `scripts/` to produce the final CSVs used by the VGAE.

### Primary Sources

| Source | What it provides | Link |
|--------|-----------------|------|
| **Sephora Products Dataset** (HuggingFace) | 8,494 real products with names, brands, prices, ratings, categories, ingredients | `MayaKitzis/sephora_products` on HuggingFace |
| **The Pudding Shade Dataset** | 6,816 foundation shades with hex colours across 107 brands | `pudding_allShades.csv` |
| **Sephora Skincare Reviews** (HuggingFace) | Real customer reviews matched by brand + product name | `eyachawechi/my-sephora-data` on HuggingFace |

### Data Pipeline

The raw datasets were processed through a multi-step pipeline:

1. **`scripts/import_sephora.py`** -- Reads `sephora_raw.csv` (8,494 rows from HuggingFace), maps Sephora categories to our 11 buckets (foundation, concealer, blush, eyeshadow, mascara, lipstick, powder, highlighter, contour, primer, perfume), extracts skin type compatibility, finish, and key ingredients. Outputs `products.csv` (1,600 curated products).

2. **`scripts/merge_shades.py`** + **`scripts/super_merge_shades.py`** -- Merges The Pudding shade dataset (6,816 shades) with additional FoundationFinder data. Cleans hex values, normalises brand/product names, and deduplicates. Outputs `shades.json` used for shade matching.

3. **`scripts/fetch_real_images.py`** + **`scripts/fetch_sephora_images.py`** -- Fetches real product photos from brand CDNs (Sephora, Amazon, brand websites) via image search. Updates `image_url` column in `products.csv`.

4. **`scripts/extract_product_colors.py`** -- Downloads each product's image and runs K-means clustering (k=6) to extract the dominant product colour (filtering out packaging, skin tones, whites, and blacks). Writes `product_hex` column used by the colour advisor for blush/lipstick/eyeshadow matching.

5. **`scripts/fill_missing_colors.py`** -- Fills remaining products missing a `product_hex` using category-based heuristics and fallback colour families.

6. **`scripts/build_realistic_reviews.py`** + **`scripts/merge_real_reviews.py`** -- Combines real Sephora reviews (matched by brand + name) with generated reviews that vary by category, finish, skin type, concerns, and rating. Produces `reviews.csv` (8,000 reviews, 5 per product).

### Final Data Files

| File | Rows | Source | Description |
|------|------|--------|-------------|
| `products.csv` | 1,600 | Sephora HuggingFace dataset, processed | Products across 11 categories with price, rating, finish, skin type, ingredients, image URLs, product hex colours |
| `interactions.csv` | 5,518 | Generated | Synthetic user-product interactions (like, view, purchase) with skin type and event type, used to build graph edges |
| `ingredients.csv` | 30 | Curated | Active ingredients (niacinamide, salicylic acid, hyaluronic acid, etc.) mapped to skin concerns they treat |
| `looks.csv` | 10 | Curated | Complete makeup looks (e.g. "Radiant Bride", "Power Office") with hand-picked product ID lists |
| `reviews.csv` | 8,000 | Mixed (real + generated) | 5 reviews per product with rating, reviewer name, title, body text |
| `shades.json` | ~4,800 | The Pudding + FoundationFinder | Foundation/concealer shades with hex colours, undertone, brand, product name |
| `sephora_raw.csv` | 8,494 | HuggingFace raw dump | Original unprocessed Sephora dataset (kept for reproducibility) |
| `pudding_allShades.csv` | 6,816 | The Pudding | Original shade dataset (kept for reproducibility) |

### Product Categories

The 1,600 products span 11 makeup categories:

| Category | Count | Special fields |
|----------|-------|----------------|
| Foundation | ~180 | shade matching via `shades.json` |
| Concealer | ~120 | shade matching |
| Blush | ~140 | `product_hex` for colour advisor |
| Eyeshadow | ~150 | `product_hex`, palette detection |
| Mascara | ~80 | -- |
| Lipstick | ~160 | `product_hex` for colour advisor |
| Powder | ~80 | -- |
| Highlighter | ~80 | `product_hex` |
| Contour | ~60 | shade matching |
| Primer | ~70 | -- |
| Perfume | 600 | `scent_family`, `is_unisex` |

---

## How to Run

### 1. Install dependencies

```bash
pip install fastapi uvicorn torch torch-geometric pandas numpy scikit-learn Pillow python-multipart
```

### 2. Train the VGAE model (optional, auto-trains on first request)

```bash
python train_vgae.py
```

This creates `models/vgae_beauty.pt` (~1 minute). The model is small (28KB) and regenerates automatically if missing.

### 3. CNN models for skin analysis (optional)

The app can optionally use two CNN classifiers for skin type and acne detection. These are pre-trained TensorFlow SavedModels (~23MB each) and are **not included in the repository** due to size.

- `models/skin_cnn/` -- classifies skin type (dry, normal, oily)
- `models/acne_cnn/` -- detects acne severity (low, moderate, severe)

**Without them**, the app falls back to heuristic analysis (which works well). If you have them, place them in the `models/` folder.

### 4. Start the server

```bash
python api.py
```

Open **http://localhost:8000** in your browser.

---

## Pages

### Main Consultation (/)

1. **Landing** -- explains what the app does, how it works, and links to perfume discovery
2. **Step 1: Occasion** -- wedding, office, night out, everyday, glam date, festival
3. **Step 2: Shade Matching** -- upload face/wrist photos for AI undertone detection
4. **Step 3: Skin Profile** -- skin type, undertone, depth (with AI suggestions from step 2)
5. **Step 4: Preferences** -- skin concerns, budget, category selection
6. **Step 5: Results** -- product cards or full routine timeline with shade matching

### Perfume Discovery (/perfumes)

- Select an occasion and/or scent families
- VGAE scores all 600 perfumes with event-based and scent family boosting
- Results show perfume cards with scent family badges, ratings, and AI-generated reasons

---

## Project Structure

```
beauty_assistant/
|-- api.py                  # FastAPI server (entry point)
|-- train_vgae.py           # Standalone VGAE training script
|-- requirements.txt
|-- README.md
|-- data/
|   |-- products.csv        # 1,600 Sephora products
|   |-- interactions.csv    # 5,518 user interactions
|   |-- ingredients.csv     # 30 active ingredients
|   |-- looks.csv           # 10 curated looks
|   |-- reviews.csv         # 8,000 reviews
|   +-- shades.json         # Foundation shade database
|-- graph/
|   |-- vgae_model.py       # VGAE architecture
|   |-- graph_builder.py    # CSV to PyG graph conversion
|   |-- trainer.py          # Training loop
|   |-- recommender.py      # Inference + content boosting
|   |-- shade_matcher.py    # Foundation shade matching
|   |-- shade_analyzer.py   # LAB undertone analysis
|   |-- face_skin_extractor.py  # MediaPipe skin extraction
|   |-- skin_heuristics.py  # Heuristic skin/acne detection
|   +-- color_advisor.py    # Colour family recommendations
|-- models/
|   +-- vgae_beauty.pt      # Trained model checkpoint
|-- static/
|   |-- index.html          # Main consultation SPA
|   +-- perfumes.html       # Perfume discovery page
+-- scripts/                # Data processing utilities
```

---

## Technologies

- **Python 3.10+**
- **PyTorch + PyTorch Geometric** -- VGAE model and graph operations
- **FastAPI + Uvicorn** -- async web server
- **MediaPipe** -- face mesh for skin tone extraction
- **Pandas / NumPy / scikit-learn** -- data processing and KNN
- **Pillow** -- image processing for shade analysis

---

## Author

**Maria Patrut** -- Master's Student in Artificial Intelligence

---

## License

This project is for educational and demonstration purposes. Product names and brands are used for reference only.
