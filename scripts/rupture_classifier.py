"""
rupture_classifier.py
=====================
Stage 2, Part 1: baseline rupture-risk classifier.

Trains a classifier to predict ruptured vs unruptured from a shape's
latent encoding (produced by a trained autoencoder). This establishes the
BASELINE accuracy using real data only — the number we later try to beat
with synthetic data augmentation.

Why classify from the latent code (not raw points)?
  - The autoencoder already compressed each shape into a meaningful
    512-d vector. A small classifier on top is fast, stable, and directly
    tests whether the learned latent space carries rupture-relevant signal.
  - It also sets up the augmentation experiment cleanly: synthetic shapes
    are just extra latent vectors with known class labels.

Handles the class imbalance (474 unruptured / 261 ruptured) with optional
class weighting so the baseline isn't trivially biased to the majority class.

Usage:
  python scripts/rupture_classifier.py \
      --processed_dir aneurysm_project/data/processed_sdf \
      --encoder_ckpt  aneurysm_project/models/vae_compare/best.pt \
      --out_dir       aneurysm_project/results_classifier
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             confusion_matrix, balanced_accuracy_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ae_models import AneurysmAE
from dataset import SurfaceDataset


# ─────────────────────────────────────────────────────────────────────────────
# Encode every shape in a split into latent vectors + labels
# ─────────────────────────────────────────────────────────────────────────────

def encode_split(model, processed_dir, split, device, n_points=2048):
    ds = SurfaceDataset(processed_dir, split=split, n_points=n_points)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    Z, y = [], []
    with torch.no_grad():
        for b in loader:
            pts = b["points"].to(device)
            mu, _ = model.encode(pts)            # use posterior mean
            Z.append(mu.cpu().numpy())
            y.extend(b["rupture_label"].numpy().tolist())
    return np.concatenate(Z), np.array(y)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier head
# ─────────────────────────────────────────────────────────────────────────────

class RuptureClassifier(nn.Module):
    """Small MLP: latent (512) -> hidden -> 1 logit."""
    def __init__(self, latent_dim=512, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Train / evaluate
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(Ztr, ytr, Zva, yva, device,
                     n_epochs=200, lr=1e-3, class_weight=True):
    model = RuptureClassifier(latent_dim=Ztr.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Class weighting for imbalance: pos_weight = n_neg / n_pos
    if class_weight:
        n_pos = max(int(ytr.sum()), 1)
        n_neg = len(ytr) - n_pos
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    else:
        pos_weight = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr = torch.from_numpy(Ztr).float().to(device)
    Ytr = torch.from_numpy(ytr).float().to(device)
    Xva = torch.from_numpy(Zva).float().to(device)

    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=32, shuffle=True)

    best_auc, best_state = 0.0, None
    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); opt.step()

        # Validation AUC
        model.eval()
        with torch.no_grad():
            va_logits = model(Xva).cpu().numpy()
        va_prob = 1 / (1 + np.exp(-va_logits))
        if len(np.unique(yva)) > 1:
            auc = roc_auc_score(yva, va_prob)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc


def evaluate(model, Z, y, device, tag=""):
    Xt = torch.from_numpy(Z).float().to(device)
    model.eval()
    with torch.no_grad():
        logits = model(Xt).cpu().numpy()
    prob = 1 / (1 + np.exp(-logits))
    pred = (prob >= 0.5).astype(int)

    res = {
        "accuracy":          float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1":                float(f1_score(y, pred, zero_division=0)),
        "auc":               float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "confusion_matrix":  confusion_matrix(y, pred).tolist(),
    }
    print(f"  [{tag}] Acc {res['accuracy']:.3f} | "
          f"BalAcc {res['balanced_accuracy']:.3f} | "
          f"F1 {res['f1']:.3f} | AUC {res['auc']:.3f}")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Load encoder
    ckpt = torch.load(args.encoder_ckpt, map_location=device)
    enc = AneurysmAE(latent_dim=ckpt["latent_dim"], hidden_dim=ckpt["hidden_dim"],
                     n_freqs=ckpt["n_freqs"], variational=ckpt["variational"],
                     beta=ckpt.get("beta", 1.0), mode_name=ckpt["mode"]).to(device)
    enc.load_state_dict(ckpt["model_state_dict"]); enc.eval()
    print(f"Encoder: {ckpt['mode']} (val recon {ckpt['val_recon']:.4f})")

    # Encode splits
    print("\nEncoding shapes to latent space...")
    Ztr, ytr = encode_split(enc, args.processed_dir, "train", device)
    Zva, yva = encode_split(enc, args.processed_dir, "val",   device)
    Zte, yte = encode_split(enc, args.processed_dir, "test",  device)
    print(f"  Train {len(ytr)} ({int(ytr.sum())} ruptured) | "
          f"Val {len(yva)} ({int(yva.sum())}) | "
          f"Test {len(yte)} ({int(yte.sum())})")

    # Train baseline classifier
    print("\nTraining baseline classifier (real data only)...")
    clf, best_val_auc = train_classifier(Ztr, ytr, Zva, yva, device,
                                         n_epochs=args.n_epochs, lr=args.lr,
                                         class_weight=not args.no_class_weight)
    print(f"  Best val AUC: {best_val_auc:.3f}")

    # Evaluate on test
    print("\nTest-set performance (BASELINE — real data only):")
    test_res = evaluate(clf, Zte, yte, device, tag="test")

    # Save baseline
    baseline = {
        "encoder_mode": ckpt["mode"],
        "n_train": int(len(ytr)), "n_train_ruptured": int(ytr.sum()),
        "n_test":  int(len(yte)), "n_test_ruptured":  int(yte.sum()),
        "best_val_auc": float(best_val_auc),
        "test": test_res,
        "class_weighting": not args.no_class_weight,
    }
    json.dump(baseline, open(out / "baseline_classifier.json", "w"), indent=2)

    # Save the encoded latents for the augmentation experiment (reuse later)
    np.savez(out / "encoded_latents.npz",
             Ztr=Ztr, ytr=ytr, Zva=Zva, yva=yva, Zte=Zte, yte=yte)

    # Confusion matrix plot
    cm = np.array(test_res["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Unruptured", "Ruptured"])
    ax.set_yticklabels(["Unruptured", "Ruptured"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Baseline Classifier (Test)\nAUC {test_res['auc']:.3f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out / "baseline_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved baseline to {out}/baseline_classifier.json")
    print(f"Saved encoded latents to {out}/encoded_latents.npz "
          f"(reused by the augmentation experiment)")
    print(f"\n>>> BASELINE TEST AUC: {test_res['auc']:.3f}  "
          f"BalAcc: {test_res['balanced_accuracy']:.3f} <<<")
    print("This is the number the synthetic-augmentation study will try to beat.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=Path, required=True)
    p.add_argument("--encoder_ckpt",  type=Path, required=True)
    p.add_argument("--out_dir",       type=Path, required=True)
    p.add_argument("--n_epochs",      type=int,   default=200)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--no_class_weight", action="store_true",
                   help="Disable class weighting (not recommended given imbalance)")
    args = p.parse_args()
    run(args)