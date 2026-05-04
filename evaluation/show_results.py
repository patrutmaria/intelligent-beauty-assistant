"""
    /opt/anaconda3/bin/python evaluation/show_results.py
"""

import os
import pandas as pd, numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(_dir, "experiments_log.csv"))

methods = [
    ("Random",             "Random"),
    ("Popularity",         "Popularity"),
    ("Pop+Boost",          "Pop+Boost"),
    ("VGAE_BPR_baseline",  "VGAE (baseline)"),
    ("Exp1_KL_anneal",     "VGAE+KL anneal"),
    ("Exp2_deterministic", "VGAE+Determ."),
    ("Exp3_multilayer_det","VGAE+MultiLayer"),
    ("Exp4_lgcn_init",     "VGAE+LGCNinit"),
    ("LightGCN_ref",       "LightGCN"),
]

def fmt(vals):
    m, s = np.mean(vals), np.std(vals)
    if m == 0: return "    —    "
    if s > 0.0001: return f"{m:.4f}±{s:.4f}"
    return f"  {m:.4f}  "

print(f"\n  {'Method':<19} {'Recall@10':>13}  {'NDCG@10':>13}  {'AUC':>13}")
print(f"  {''*19} {''*13}  {''*13}  {''*13}")
for key, label in methods:
    g = df[df.experiment == key]
    print(f"  {label:<19} {fmt(g['Recall@10'].values):>13}  {fmt(g['NDCG@10'].values):>13}  {fmt(g['Val_AUC'].values):>13}")
print()
