"""
augmentation_experiment.py
==========================
Stage 2, Part 3: the headline experiment.

Tests whether synthetic data augmentation with the CVAE improves the
rupture-risk classifier — especially on the minority (ruptured) class.

Procedure:
  1. Load the real encoded latents (from rupture_classifier.py baseline).
  2. Use the trained CVAE to generate synthetic RUPTURED shapes, encode
     them to latent vectors, and add them to the training set to rebalance
     the 189/325 ruptured/unruptured training split toward 50/50.
  3. Retrain the SAME classifier architecture on the augmented set.
  4. Compare against the real-only baseline on the IDENTICAL real test set.

The test set is never augmented — only real shapes — so the comparison is
fair: we measure whether synthetic training data improves generalisation to
real data.

Usage:
  python scripts/augmentation_experiment.py \
      --processed_dir aneurysm_project/data/processed_sdf \
      --cvae_ckpt     aneurysm_project/models/cvae/best_cvae.pt \
      --encoder_ckpt  aneurysm_project/models/vae_compare/best.pt \
      --baseline_dir  aneurysm_project/results_classifier \
      --out_dir       aneurysm_project/results_augmentation \
      --n_synthetic   200
"""

import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             balanced_accuracy_score, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cvae_model import AneurysmCVAE
from ae_models import AneurysmAE
from rupture_classifier import RuptureClassifier, train_classifier, evaluate


def make_query_grid(res, device):
    lin = np.linspace(-1, 1, res)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], -1).astype(np.float32)
    return torch.from_numpy(pts).to(device)


def cvae_sdf_grid(cvae, label, res, device, batch=65536):
    """Generate one shape of `label`, return its SDF grid."""
    grid = make_query_grid(res, device)
    z = torch.randn(1, cvae.latent_dim, device=device)
    lab = torch.tensor([label], dtype=torch.long, device=device)
    vals = []
    with torch.no_grad():
        for i in range(0, len(grid), batch):
            chunk = grid[i:i+batch].unsqueeze(0)
            vals.append(cvae.decode(chunk, z, lab).squeeze().cpu().numpy())
    return np.concatenate(vals).reshape(res, res, res), grid.cpu().numpy()


def synth_surface_from_grid(sdf_grid, grid_np, n_pts=2048):
    """Extract a near-surface point cloud from a generated SDF grid."""
    sdf = sdf_grid.ravel()
    near = np.abs(sdf) < 0.03
    pts = grid_np[near]
    if len(pts) == 0:
        pts = grid_np[np.abs(sdf) < 0.08]
    if len(pts) == 0:
        return None
    idx = np.random.choice(len(pts), n_pts, replace=(len(pts) < n_pts))
    return pts[idx].astype(np.float32)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # ── Load real encoded latents from the baseline ───────────────────────
    data = np.load(Path(args.baseline_dir) / "encoded_latents.npz")
    Ztr, ytr = data["Ztr"], data["ytr"]
    Zva, yva = data["Zva"], data["yva"]
    Zte, yte = data["Zte"], data["yte"]
    print(f"Real train: {len(ytr)} ({int(ytr.sum())} ruptured)")
    print(f"Real test : {len(yte)} ({int(yte.sum())} ruptured)")

    # ── Load CVAE (generator) and encoder (to embed synthetic shapes) ─────
    cck = torch.load(args.cvae_ckpt, map_location=device)
    cvae = AneurysmCVAE(latent_dim=cck["latent_dim"], hidden_dim=cck["hidden_dim"],
                        n_freqs=cck["n_freqs"]).to(device)
    cvae.load_state_dict(cck["model_state_dict"]); cvae.eval()

    eck = torch.load(args.encoder_ckpt, map_location=device)
    enc = AneurysmAE(latent_dim=eck["latent_dim"], hidden_dim=eck["hidden_dim"],
                     n_freqs=eck["n_freqs"], variational=eck["variational"],
                     beta=eck.get("beta",1.0), mode_name=eck["mode"]).to(device)
    enc.load_state_dict(eck["model_state_dict"]); enc.eval()

    # ── Generate synthetic RUPTURED shapes to rebalance ───────────────────
    # Train set has 189 ruptured / 325 unruptured. Generate ruptured to balance.
    n_needed = max(args.n_synthetic, 325 - 189)
    print(f"\nGenerating {n_needed} synthetic ruptured shapes...")
    res = args.resolution
    syn_Z, syn_y = [], []
    made = 0
    for i in range(n_needed * 2):  # over-attempt; some extractions may fail
        if made >= n_needed:
            break
        sdf_grid, grid_np = cvae_sdf_grid(cvae, label=1, res=res, device=device)
        pts = synth_surface_from_grid(sdf_grid, grid_np)
        if pts is None:
            continue
        pts_t = torch.from_numpy(pts).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, _ = enc.encode(pts_t)            # embed with the SAME encoder
        syn_Z.append(mu.cpu().numpy()[0]); syn_y.append(1)
        made += 1
        if made % 25 == 0:
            print(f"  {made}/{n_needed}")

    syn_Z = np.array(syn_Z); syn_y = np.array(syn_y)
    print(f"Generated {len(syn_y)} synthetic ruptured latents")

    # ── Build augmented training set ──────────────────────────────────────
    Ztr_aug = np.concatenate([Ztr, syn_Z], axis=0)
    ytr_aug = np.concatenate([ytr, syn_y], axis=0)
    print(f"Augmented train: {len(ytr_aug)} "
          f"({int(ytr_aug.sum())} ruptured, {len(ytr_aug)-int(ytr_aug.sum())} unruptured)")

    # ── Train classifier on augmented set (no class weight — now balanced) ─
    print("\nTraining classifier on AUGMENTED data...")
    clf_aug, val_auc_aug = train_classifier(Ztr_aug, ytr_aug, Zva, yva, device,
                                            n_epochs=args.n_epochs, lr=1e-3,
                                            class_weight=False)
    print("\nAUGMENTED test performance:")
    aug_res = evaluate(clf_aug, Zte, yte, device, tag="augmented")

    # ── Reload baseline result for comparison ─────────────────────────────
    baseline = json.load(open(Path(args.baseline_dir) / "baseline_classifier.json"))
    base_res = baseline["test"]

    # ── Compare ───────────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print("AUGMENTATION RESULT — Baseline (real only) vs Augmented")
    print(f"{'='*56}")
    print(f"  {'Metric':<20} {'Baseline':>12} {'Augmented':>12} {'Δ':>8}")
    print(f"  {'-'*52}")
    for met in ["auc", "balanced_accuracy", "f1", "accuracy"]:
        b, a = base_res[met], aug_res[met]
        print(f"  {met:<20} {b:>12.3f} {a:>12.3f} {a-b:>+8.3f}")

    comparison = {
        "baseline": base_res, "augmented": aug_res,
        "n_synthetic": int(len(syn_y)),
        "deltas": {m: aug_res[m] - base_res[m]
                   for m in ["auc","balanced_accuracy","f1","accuracy"]},
    }
    json.dump(comparison, open(out / "augmentation_comparison.json","w"), indent=2)

    # ── Bar chart ─────────────────────────────────────────────────────────
    mets = ["auc", "balanced_accuracy", "f1", "accuracy"]
    labels = ["AUC", "Balanced\nAcc", "F1\n(ruptured)", "Accuracy"]
    bx = np.arange(len(mets)); w = 0.36
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(bx - w/2, [base_res[m] for m in mets], w, label="Baseline (real only)", color="#888888")
    ax.bar(bx + w/2, [aug_res[m]  for m in mets], w, label="Augmented (+ synthetic)", color="#1A8754")
    ax.set_xticks(bx); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Rupture Classifier: Real-only vs Synthetic-Augmented")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    for i, m in enumerate(mets):
        ax.text(i - w/2, base_res[m]+0.02, f"{base_res[m]:.2f}", ha="center", fontsize=9)
        ax.text(i + w/2, aug_res[m]+0.02,  f"{aug_res[m]:.2f}",  ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "augmentation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved: augmentation_comparison.png, augmentation_comparison.json")
    d_f1  = aug_res["f1"] - base_res["f1"]
    d_bal = aug_res["balanced_accuracy"] - base_res["balanced_accuracy"]
    print(f"\n>>> F1 (ruptured) change: {d_f1:+.3f} | "
          f"Balanced accuracy change: {d_bal:+.3f} <<<")
    if d_f1 > 0 or d_bal > 0:
        print("Synthetic augmentation improved minority-class performance.")
    else:
        print("No improvement — worth analysing why (synthetic shape quality, "
              "latent realism, or volume bias affecting embeddings).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=Path, required=True)
    p.add_argument("--cvae_ckpt",     type=Path, required=True)
    p.add_argument("--encoder_ckpt",  type=Path, required=True)
    p.add_argument("--baseline_dir",  type=Path, required=True)
    p.add_argument("--out_dir",       type=Path, required=True)
    p.add_argument("--n_synthetic",   type=int, default=200)
    p.add_argument("--resolution",    type=int, default=64)
    p.add_argument("--n_epochs",      type=int, default=200)
    args = p.parse_args()
    run(args)