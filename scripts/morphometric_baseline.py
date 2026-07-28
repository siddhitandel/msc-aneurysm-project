"""
morphometric_baseline.py
========================
Classical (non-deep) rupture-risk baseline using AneuX's own published
morphometric indices (the `gi` block of morpho-per-cut.csv).

Purpose: answer the question an examiner will ask —
  "Is your deep generative/latent approach actually better than classical
   morphometric shape analysis?"

This trains standard classifiers (logistic regression, random forest) on the
12 geometric indices that the aneurysm-rupture literature uses:
  Shape: AR, BF, CP, EI, NSI, UI
  Size : Dmax, Dn, H, S, V, aSz
and reports the SAME metrics (AUC, balanced accuracy, F1) as the deep
latent-space classifier, on the SAME train/test split, so the two are
directly comparable.

Usage:
  python scripts/morphometric_baseline.py \
      --morpho_csv  ~/MSc_Project/6678442/data/morpho-per-cut.csv \
      --clinical_csv ~/MSc_Project/6678442/data/clinical.csv \
      --processed_dir aneurysm_project/data/processed_sdf \
      --out_dir aneurysm_project/results_morphometric

The --processed_dir is used only to reproduce the identical train/val/test
split (via SurfaceDataset), so the comparison to the deep model is fair.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             f1_score, accuracy_score, confusion_matrix)

from dataset import SurfaceDataset


GI_FEATURES = [
    "gi__shape__AR", "gi__shape__BF", "gi__shape__CP",
    "gi__shape__EI", "gi__shape__NSI", "gi__shape__UI",
    "gi__size__Dmax", "gi__size__Dn", "gi__size__H",
    "gi__size__S", "gi__size__V", "gi__size__aSz",
]


def load_morphometrics(morpho_csv):
    """Load morpho-per-cut.csv, filter to dome cut, return dataset-indexed gi features."""
    df = pd.read_csv(morpho_csv, header=[0, 1, 2])
    df = df.iloc[1:].reset_index(drop=True)          # drop embedded sub-header row
    df.columns = ["__".join(str(x) for x in c) for c in df.columns]

    ds_col  = df.columns[1]    # dataset ID
    cut_col = df.columns[2]    # cutType

    dome = df[df[cut_col] == "dome"].copy()
    dome = dome.rename(columns={ds_col: "dataset"})

    keep = ["dataset"] + GI_FEATURES
    dome = dome[keep].copy()
    for c in GI_FEATURES:
        dome[c] = pd.to_numeric(dome[c], errors="coerce")
    dome = dome.dropna(subset=GI_FEATURES)
    return dome.set_index("dataset")


def load_labels(clinical_csv):
    """Return dict: dataset -> rupture label (1 ruptured / 0 unruptured)."""
    cl = pd.read_csv(clinical_csv)
    # Identify the dataset and status columns robustly
    ds_col = "dataset" if "dataset" in cl.columns else cl.columns[0]
    status_col = None
    for c in cl.columns:
        if "status" in c.lower() or "ruptur" in c.lower():
            status_col = c
            break
    if status_col is None:
        raise ValueError(f"Could not find a rupture status column in {clinical_csv}")

    cl = cl[[ds_col, status_col]].dropna()
    mapping = {}
    for _, row in cl.iterrows():
        s = str(row[status_col]).strip().lower()
        if s in ("ruptured", "1", "true", "yes"):
            mapping[str(row[ds_col])] = 1
        elif s in ("unruptured", "0", "false", "no"):
            mapping[str(row[ds_col])] = 0
    return mapping


def split_ids(processed_dir):
    """Reproduce the exact same train/val/test IDs used by the deep model."""
    splits = {}
    for split in ["train", "val", "test"]:
        ds = SurfaceDataset(processed_dir, split=split, n_points=2048)
        splits[split] = list(ds.ids)
    return splits


def evaluate(model, X, y, tag):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)
    res = {
        "auc":               float(roc_auc_score(y, prob)) if len(set(y)) > 1 else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1":                float(f1_score(y, pred, zero_division=0)),
        "accuracy":          float(accuracy_score(y, pred)),
        "confusion_matrix":  confusion_matrix(y, pred).tolist(),
    }
    print(f"  [{tag}] AUC {res['auc']:.3f} | BalAcc {res['balanced_accuracy']:.3f} | "
          f"F1 {res['f1']:.3f} | Acc {res['accuracy']:.3f}")
    return res


def run(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("Loading morphometrics (dome cut)...")
    morph = load_morphometrics(args.morpho_csv)
    print(f"  {len(morph)} aneurysms with gi features")

    print("Loading rupture labels...")
    labels = load_labels(args.clinical_csv)
    print(f"  {len(labels)} labelled aneurysms")

    print("Reproducing deep-model train/val/test split...")
    splits = split_ids(args.processed_dir)

    # Build feature matrices per split
    def build(split_ids_list):
        X, y, kept = [], [], []
        for sid in split_ids_list:
            if sid in morph.index and sid in labels:
                X.append(morph.loc[sid, GI_FEATURES].values.astype(float))
                y.append(labels[sid]); kept.append(sid)
        return np.array(X), np.array(y), kept

    Xtr, ytr, _ = build(splits["train"])
    Xva, yva, _ = build(splits["val"])
    Xte, yte, _ = build(splits["test"])
    print(f"  Train {len(ytr)} ({int(ytr.sum())} ruptured) | "
          f"Val {len(yva)} ({int(yva.sum())}) | "
          f"Test {len(yte)} ({int(yte.sum())})")

    # Standardise features (fit on train only)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    results = {}

    # ── Logistic Regression (class-weighted for imbalance) ────────────────
    print("\nLogistic Regression (balanced):")
    lr = LogisticRegression(class_weight="balanced", max_iter=2000)
    lr.fit(Xtr_s, ytr)
    results["logreg"] = {
        "val":  evaluate(lr, Xva_s, yva, "val"),
        "test": evaluate(lr, Xte_s, yte, "test"),
    }

    # ── Random Forest ─────────────────────────────────────────────────────
    print("\nRandom Forest (balanced):")
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                max_depth=6, random_state=42)
    rf.fit(Xtr_s, ytr)
    results["random_forest"] = {
        "val":  evaluate(rf, Xva_s, yva, "val"),
        "test": evaluate(rf, Xte_s, yte, "test"),
    }

    # ── Feature importance (random forest) ────────────────────────────────
    importances = dict(zip([f.split("__")[-1] for f in GI_FEATURES],
                           rf.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda kv: -kv[1]))
    results["rf_feature_importance"] = importances

    print("\nTop morphometric predictors (random forest importance):")
    for k, v in list(importances.items())[:6]:
        print(f"  {k:6s}: {v:.3f}")

    json.dump(results, open(out / "morphometric_baseline.json", "w"), indent=2)

    # ── Plot: feature importance + comparison-ready bar ───────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(importances.keys()); vals = list(importances.values())
    ax.barh(names[::-1], vals[::-1], color="#1C7293")
    ax.set_xlabel("Random Forest Importance")
    ax.set_title("Morphometric Feature Importance for Rupture Prediction")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out / "morphometric_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary
    best = max(["logreg", "random_forest"],
               key=lambda m: results[m]["test"]["auc"])
    print(f"\n{'='*56}")
    print("MORPHOMETRIC BASELINE (classical) — test set")
    print(f"{'='*56}")
    print(f"  Logistic Regression : AUC {results['logreg']['test']['auc']:.3f}")
    print(f"  Random Forest       : AUC {results['random_forest']['test']['auc']:.3f}")
    print(f"\n  Best classical AUC  : {results[best]['test']['auc']:.3f} ({best})")
    print(f"  (Deep latent-space classifier baseline was AUC 0.724)")
    print(f"\nSaved: morphometric_baseline.json, morphometric_importance.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--morpho_csv",   type=Path, required=True)
    p.add_argument("--clinical_csv", type=Path, required=True)
    p.add_argument("--processed_dir", type=Path, required=True)
    p.add_argument("--out_dir",      type=Path, required=True)
    args = p.parse_args()
    run(args)