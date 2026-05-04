"""
Generate synthetic user–product interactions for VGAE training.

Strategy (reverse-engineered from existing data/interactions.csv):
  - 200 users, each with 20-35 random interactions (uniform)
  - Products sampled uniformly (no rating/price bias) — proportional to category size
  - No duplicate (user, product) pairs
  - Interaction types: ~49% view, ~32% like, ~19% purchase (assigned randomly)
  - Each user gets a random skin_type and event_type
  - Skin types:  oily/combination/dry/normal  (roughly uniform)
  - Event types: festival/glam/office/wedding/evening/natural

This produces a cold-start-friendly dataset where signal comes from the graph
structure (product–look, product–ingredient edges), not from interaction density.

Usage:
    /opt/anaconda3/bin/python scripts/generate_interactions.py
    /opt/anaconda3/bin/python scripts/generate_interactions.py --users 200 --seed 42
"""

import argparse
import os
import numpy as np
import pandas as pd


SKIN_TYPES = ["oily", "dry", "combination", "normal"]
EVENT_TYPES = ["wedding", "office", "natural", "evening", "glam", "festival"]

INTERACTION_WEIGHTS = {"view": 0.49, "like": 0.32, "purchase": 0.19}


def generate(products_csv: str, output_csv: str, n_users: int = 200,
             min_inter: int = 20, max_inter: int = 35, seed: int = 42):
    rng = np.random.RandomState(seed)

    products = pd.read_csv(products_csv)
    all_pids = products["product_id"].tolist()

    interaction_types = list(INTERACTION_WEIGHTS.keys())
    interaction_probs = list(INTERACTION_WEIGHTS.values())

    rows = []

    for uid in range(1, n_users + 1):
        skin = rng.choice(SKIN_TYPES)
        event = rng.choice(EVENT_TYPES)
        n_interactions = rng.randint(min_inter, max_inter + 1)

        # Sample products uniformly without replacement
        sampled_pids = rng.choice(all_pids, size=min(n_interactions, len(all_pids)),
                                  replace=False)

        for pid in sampled_pids:
            itype = rng.choice(interaction_types, p=interaction_probs)
            rows.append({
                "user_id": uid,
                "product_id": int(pid),
                "interaction_type": itype,
                "skin_type": skin,
                "event_type": event,
            })

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Stats
    print(f"Generated {len(df)} interactions for {n_users} users")
    print(f"  Products in catalog: {len(all_pids)}")
    print(f"  Interactions/user: {df.groupby('user_id').size().min()}-{df.groupby('user_id').size().max()} "
          f"(mean {df.groupby('user_id').size().mean():.1f})")
    print(f"  Types: {dict(df.interaction_type.value_counts())}")
    print(f"  Skin types: {dict(df.drop_duplicates('user_id').skin_type.value_counts())}")
    print(f"  Event types: {dict(df.drop_duplicates('user_id').event_type.value_counts())}")
    print(f"  Saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic interactions")
    parser.add_argument("--products", default="data/products.csv")
    parser.add_argument("--output", default="data/interactions.csv")
    parser.add_argument("--users", type=int, default=200)
    parser.add_argument("--min_inter", type=int, default=20)
    parser.add_argument("--max_inter", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(args.products, args.output, args.users, args.min_inter, args.max_inter, args.seed)
