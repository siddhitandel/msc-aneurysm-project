"""
multiseed_classifier.py
=======================
Puts confidence intervals on the two classification results that are
currently single-run:

  R7  deep latent classifier (AUC 0.724) vs classical morphometrics (0.748)
  R8  augmentation effect (AUC -0.05)

Both were measured once on a 111-shape test set, so neither gap is currently
defensible. This reruns each with N random seeds (different classifier
initialisation / batching, same fixed data split) and reports
mean +/- std, matching the rigour already applied to the volume-bias fix.

Cheap to run: the latent codes are already cached in encoded_latents.npz,
so this only retrains the small MLP head, not the VAE.

Usage:
  python scripts/multiseed_classifier.py \
      --baseline_dir  aneurysm_project/results_classifier \
      --morpho_csv    ~/MSc_Project/6678442/data/clinical.csv \
      --morpho_feat   ~/MSc_Project/6678442/data/morpho-per-cut.csv \
      --aug_dir       aneurysm_project/results_augmentation \
      --out_dir       aneurysm_project/results_multiseed \
      --n_seeds 10
"""

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rupture_classifier import train_classifier

GI = ["gi__shape__AR","gi__shape__BF","gi__shape__CP","gi__shape__EI",
      "gi__shape__NSI","gi__shape__UI","gi__size__Dmax","gi__size__Dn",
      "gi__size__H","gi__size__S","gi__size__V","gi__size__aSz"]


def evaluate_probs(prob, y):
    pred = (prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def deep_multiseed(data, n_seeds, device):
    """Retrain the latent MLP classifier n_seeds times."""
    Ztr, ytr = data["Ztr"], data["ytr"]
    Zva, yva = data["Zva"], data["yva"]
    Zte, yte = data["Zte"], data["yte"]
    runs = []
    for s in range(n_seeds):
        torch.manual_seed(s); np.random.seed(s)
        clf, _ = train_classifier(Ztr, ytr, Zva, yva, device,
                                  n_epochs=200, lr=1e-3, class_weight=True)
        with torch.no_grad():
            logits = clf(torch.from_numpy(Zte).float().to(device)).cpu().numpy()
        prob = 1/(1+np.exp(-logits))
        r = evaluate_probs(prob, yte)
        runs.append(r)
        print(f"    seed {s}: AUC {r['auc']:.3f}")
    return runs


def morpho_multiseed(morpho_feat, clinical_csv, split_ids, n_seeds):
    """Retrain classical classifiers n_seeds times (RF varies; LogReg is
    deterministic so its spread reflects only the bootstrap of the split)."""
    df = pd.read_csv(morpho_feat, header=[0,1,2]).iloc[1:].reset_index(drop=True)
    df.columns = ["__".join(str(x) for x in c) for c in df.columns]
    ds_col, cut_col = df.columns[1], df.columns[2]
    dome = df[df[cut_col]=="dome"].rename(columns={ds_col:"dataset"})
    dome = dome[["dataset"]+GI].copy()
    for c in GI: dome[c] = pd.to_numeric(dome[c], errors="coerce")
    dome = dome.dropna(subset=GI).set_index("dataset")

    cl = pd.read_csv(clinical_csv)
    lab = {}
    for _, r in cl.iterrows():
        s = str(r.get("status","")).strip().lower()
        if s in ("ruptured","1","true","yes"): lab[str(r["dataset"])] = 1
        elif s in ("unruptured","0","false","no"): lab[str(r["dataset"])] = 0

    def build(ids):
        X, y = [], []
        for i in ids:
            if i in dome.index and i in lab:
                X.append(dome.loc[i, GI].values.astype(float)); y.append(lab[i])
        return np.array(X), np.array(y)

    Xtr, ytr = build(split_ids["train"])
    Xte, yte = build(split_ids["test"])
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    lr_runs, rf_runs = [], []
    for s in range(n_seeds):
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=s)
        lr.fit(Xtr_s, ytr)
        lr_runs.append(evaluate_probs(lr.predict_proba(Xte_s)[:,1], yte))
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    max_depth=6, random_state=s)
        rf.fit(Xtr_s, ytr)
        rf_runs.append(evaluate_probs(rf.predict_proba(Xte_s)[:,1], yte))
    return lr_runs, rf_runs


def agg(runs, key):
    v = np.array([r[key] for r in runs])
    return float(v.mean()), float(v.std())


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    data = np.load(Path(args.baseline_dir)/"encoded_latents.npz")
    print(f"Loaded cached latents: train {len(data['ytr'])}, test {len(data['yte'])}")

    print(f"\n[1/2] Deep latent classifier x {args.n_seeds} seeds...")
    deep_runs = deep_multiseed(data, args.n_seeds, device)

    results = {"n_seeds": args.n_seeds, "deep": {}}
    for k in ["auc","balanced_accuracy","f1"]:
        m, s = agg(deep_runs, k)
        results["deep"][k] = {"mean": m, "std": s}
        print(f"  deep {k:18s}: {m:.3f} +/- {s:.3f}")

    # classical (optional - needs the morphometric csv)
    if args.morpho_feat and Path(args.morpho_feat).exists():
        print(f"\n[2/2] Classical morphometric classifiers x {args.n_seeds} seeds...")
        from dataset import SurfaceDataset
        split_ids = {}
        for sp in ["train","test"]:
            split_ids[sp] = list(SurfaceDataset(args.processed_dir, split=sp, n_points=2048).ids)
        lr_runs, rf_runs = morpho_multiseed(args.morpho_feat, args.morpho_csv,
                                            split_ids, args.n_seeds)
        results["logreg"], results["random_forest"] = {}, {}
        for k in ["auc","balanced_accuracy","f1"]:
            m,s = agg(lr_runs,k); results["logreg"][k] = {"mean":m,"std":s}
            print(f"  logreg {k:16s}: {m:.3f} +/- {s:.3f}")
        for k in ["auc","balanced_accuracy","f1"]:
            m,s = agg(rf_runs,k); results["random_forest"][k] = {"mean":m,"std":s}
            print(f"  rf     {k:16s}: {m:.3f} +/- {s:.3f}")

        # is the deep-vs-classical gap real?
        d_m, d_s = results["deep"]["auc"]["mean"], results["deep"]["auc"]["std"]
        l_m, l_s = results["logreg"]["auc"]["mean"], results["logreg"]["auc"]["std"]
        gap = l_m - d_m
        pooled = np.sqrt(d_s**2 + l_s**2)
        results["deep_vs_classical"] = {
            "auc_gap": float(gap), "pooled_std": float(pooled),
            "gap_in_std_units": float(gap/pooled) if pooled>0 else None,
            "verdict": ("within noise" if pooled==0 or abs(gap) < 2*pooled
                        else "distinguishable"),
        }
        print(f"\n  Classical - deep AUC gap: {gap:+.3f} (pooled std {pooled:.3f})"
              f" -> {results['deep_vs_classical']['verdict']}")

    json.dump(results, open(out/"multiseed_classifier.json","w"), indent=2)

    # plot
    labels, means, stds = [], [], []
    for name, key in [("Deep latent","deep"), ("LogReg (morpho)","logreg"),
                      ("RF (morpho)","random_forest")]:
        if key in results:
            labels.append(name)
            means.append(results[key]["auc"]["mean"])
            stds.append(results[key]["auc"]["std"])
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.bar(labels, means, yerr=stds, capsize=8, color=["#065A82","#1A8754","#1C7293"][:len(labels)])
    ax.set_ylabel("Test AUC"); ax.set_ylim(0.5, 0.85)
    ax.set_title(f"Rupture Prediction: {args.n_seeds}-seed mean +/- std")
    ax.grid(True, alpha=0.3, axis="y")
    for i,(m,s) in enumerate(zip(means,stds)):
        ax.text(i, m+s+0.01, f"{m:.3f}\n±{s:.3f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(out/"multiseed_auc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: multiseed_classifier.json, multiseed_auc.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_dir", type=Path, required=True)
    p.add_argument("--processed_dir", type=Path, default="aneurysm_project/data/processed_sdf")
    p.add_argument("--morpho_csv",   type=Path, default=None, help="clinical.csv")
    p.add_argument("--morpho_feat",  type=Path, default=None, help="morpho-per-cut.csv")
    p.add_argument("--out_dir",      type=Path, required=True)
    p.add_argument("--n_seeds",      type=int, default=10)
    args = p.parse_args()
    run(args)