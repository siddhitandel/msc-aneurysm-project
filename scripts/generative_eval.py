"""
generative_eval.py
==================
Fidelity / Coverage / Generalization evaluation for the VAE generator,
answering the supervisor's questions:

  - "On what basis are we comparing generated vs real?"
       -> a principled set of set-to-set metrics (MMD, COV, 1-NNA), not just
          a per-shape nearest-real distance.
  - "Shouldn't seem like copies / how much does it memorize?"
       -> a memorization check: nearest-neighbour distance from each
          generated shape to the TRAINING set, compared against the
          real train-test nearest-neighbour distances. If generated shapes
          are no closer to training shapes than held-out real test shapes
          are, the model is generalizing, not copying.

Metrics (all use Chamfer distance between shape point clouds):
  MMD  (Minimum Matching Distance) - FIDELITY. For each real shape, distance
        to its nearest generated shape; averaged. Lower = generated shapes
        are realistic.
  COV  (Coverage) - COVERAGE/DIVERSITY. Fraction of real shapes that are the
        nearest neighbour of at least one generated shape. Higher = generated
        set spans the real variety (not mode-collapsed).
  1-NNA (1-Nearest-Neighbour Accuracy) - JOINT fidelity+coverage. Train a
        1-NN classifier to tell real from generated; report its leave-one-out
        accuracy. 50% is ideal (indistinguishable); >50% = distinguishable
        (poor), <50% = generated over-smoothed/duplicated.

Memorization / novelty:
  For each generated shape, nearest-neighbour Chamfer distance to the TRAIN
  set (NNN-train) and to the TEST set (NNN-test). Compared against the
  real-to-real baseline. A generated shape whose NNN-train is far below the
  typical real-real spacing is a suspected memorized copy.

Usage:
  python scripts/generative_eval.py \
      --processed_dir aneurysm_project/data/processed_sdf \
      --vae_ckpt      aneurysm_project/models/vae_compare/best.pt \
      --out_dir       aneurysm_project/results_genereval \
      --n_gen 100 --offset 0.01
"""

import argparse, json
from pathlib import Path

import numpy as np
import torch
import trimesh
from skimage import measure
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ae_models import AneurysmAE
from dataset import SurfaceDataset


# ---- geometry helpers -------------------------------------------------------

def eval_sdf_grid(model, z, res, device, batch=65536):
    lin = np.linspace(-1, 1, res)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], -1).astype(np.float32)
    pts_t = torch.from_numpy(pts).to(device)
    vals = []
    with torch.no_grad():
        for i in range(0, len(pts_t), batch):
            chunk = pts_t[i:i+batch].unsqueeze(0)
            zb = z if z.dim() == 2 else z.unsqueeze(0)
            vals.append(model.decode(chunk, zb).squeeze().cpu().numpy())
    return np.concatenate(vals).reshape(res, res, res)


def grid_to_points(sdf_grid, level, n_pts=2048):
    """Marching cubes -> sampled surface point cloud (for Chamfer)."""
    try:
        verts, faces, _, _ = measure.marching_cubes(sdf_grid, level=level)
    except (ValueError, RuntimeError):
        return None
    res = sdf_grid.shape[0]
    verts = verts / (res - 1) * 2.0 - 1.0
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if len(m.faces) == 0:
        return None
    pts, _ = trimesh.sample.sample_surface(m, n_pts)
    return pts.astype(np.float32)


def chamfer(a, b):
    """Symmetric Chamfer distance between two point clouds."""
    ta, tb = cKDTree(a), cKDTree(b)
    da, _ = tb.query(a)
    db, _ = ta.query(b)
    return float(da.mean() + db.mean())


def pairwise_chamfer(setA, setB):
    """Matrix of Chamfer distances, |A| x |B|."""
    M = np.zeros((len(setA), len(setB)))
    for i, a in enumerate(setA):
        for j, b in enumerate(setB):
            M[i, j] = chamfer(a, b)
    return M


# ---- metrics ----------------------------------------------------------------

def compute_mmd_cov(gen, real):
    """MMD (fidelity) and COV (coverage) from the gen x real Chamfer matrix."""
    D = pairwise_chamfer(gen, real)           # (n_gen, n_real)
    # MMD: for each real, nearest gen; average
    mmd = float(D.min(axis=0).mean())
    # COV: fraction of real shapes that are some gen's nearest neighbour
    nearest_real_for_each_gen = D.argmin(axis=1)
    cov = len(set(nearest_real_for_each_gen.tolist())) / len(real)
    return mmd, cov, D


def compute_1nna(gen, real):
    """1-NNA: leave-one-out 1-NN real-vs-generated classifier accuracy."""
    allshapes = list(gen) + list(real)
    labels = np.array([0]*len(gen) + [1]*len(real))   # 0=gen, 1=real
    n = len(allshapes)
    D = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i+1, n):
            d = chamfer(allshapes[i], allshapes[j])
            D[i, j] = d; D[j, i] = d
    preds = []
    for i in range(n):
        nn = np.argmin(D[i])
        preds.append(labels[nn])
    preds = np.array(preds)
    acc = float((preds == labels).mean())
    return acc


# ---- main -------------------------------------------------------------------

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    res = args.resolution

    ck = torch.load(args.vae_ckpt, map_location=device)
    model = AneurysmAE(latent_dim=ck["latent_dim"], hidden_dim=ck["hidden_dim"],
                       n_freqs=ck["n_freqs"], variational=ck["variational"],
                       beta=ck.get("beta", 1.0), mode_name=ck["mode"]).to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    print(f"Model: VAE (val recon {ck['val_recon']:.4f})")

    # Real TRAIN and TEST point clouds (from stored surface samples)
    def load_real(split, n):
        ds = SurfaceDataset(args.processed_dir, split=split, n_points=2048)
        out_pts = []
        for sid in ds.ids[:n]:
            p = np.load(Path(args.processed_dir)/"surface"/f"{sid}.npz")["points"]
            if len(p) > 2048:
                p = p[np.random.choice(len(p), 2048, replace=False)]
            out_pts.append(p.astype(np.float32))
        return out_pts

    print("Loading real train/test shapes...")
    real_train = load_real("train", args.n_ref)
    real_test  = load_real("test",  args.n_ref)
    print(f"  {len(real_train)} train, {len(real_test)} test reference shapes")

    # Generate
    print(f"Generating {args.n_gen} shapes...")
    gen = []
    tries = 0
    while len(gen) < args.n_gen and tries < args.n_gen*3:
        tries += 1
        z = torch.randn(1, model.latent_dim, device=device)
        pts = grid_to_points(eval_sdf_grid(model, z, res, device), args.offset)
        if pts is not None:
            gen.append(pts)
    print(f"  {len(gen)} valid generated shapes ({tries} attempts)")

    # ---- Fidelity + Coverage (generated vs real TEST) ----
    print("\nComputing MMD / COV (vs real test set)...")
    mmd, cov, D_gen_test = compute_mmd_cov(gen, real_test)
    print(f"  MMD (fidelity, lower=better): {mmd:.4f}")
    print(f"  COV (coverage, higher=better): {cov*100:.1f}%")

    # ---- 1-NNA (joint) ----
    print("Computing 1-NNA (vs real test set)...")
    nna = compute_1nna(gen, real_test)
    print(f"  1-NNA (ideal=50%): {nna*100:.1f}%")

    # ---- Memorization / novelty ----
    print("Computing memorization check (gen -> train vs gen -> test)...")
    D_gen_train = pairwise_chamfer(gen, real_train)
    nnn_train = D_gen_train.min(axis=1)      # each gen's nearest TRAIN shape
    nnn_test  = D_gen_test.min(axis=1)       # each gen's nearest TEST shape

    # real-real baseline: test shape -> nearest train shape (natural spacing)
    D_test_train = pairwise_chamfer(real_test, real_train)
    real_spacing = D_test_train.min(axis=1)

    mem_ratio = float(np.median(nnn_train) / (np.median(real_spacing) + 1e-9))
    n_suspect = int((nnn_train < real_spacing.min()).sum())
    print(f"  Median gen->train NN distance: {np.median(nnn_train):.4f}")
    print(f"  Median real test->train NN distance (baseline): {np.median(real_spacing):.4f}")
    print(f"  Memorization ratio (gen/real, ~1 = healthy): {mem_ratio:.2f}")
    print(f"  Suspected memorized copies (closer than any real pair): {n_suspect}/{len(gen)}")

    summary = {
        "n_gen": len(gen), "n_ref": args.n_ref, "offset": args.offset,
        "fidelity_MMD": mmd,
        "coverage_COV_pct": cov*100,
        "joint_1NNA_pct": nna*100,
        "memorization": {
            "median_gen_to_train": float(np.median(nnn_train)),
            "median_gen_to_test":  float(np.median(nnn_test)),
            "median_real_test_to_train": float(np.median(real_spacing)),
            "memorization_ratio": mem_ratio,
            "suspected_copies": n_suspect,
        },
    }
    json.dump(summary, open(out/"generative_eval.json","w"), indent=2)

    # ---- Plot: memorization histogram ----
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(real_spacing, bins=20, alpha=0.5, label="real test -> train (baseline)", color="#888888")
    ax.hist(nnn_train, bins=20, alpha=0.55, label="generated -> train", color="#C0392B")
    ax.axvline(np.median(real_spacing), color="#555555", linestyle="--", linewidth=1)
    ax.axvline(np.median(nnn_train), color="#C0392B", linestyle="--", linewidth=1)
    ax.set_xlabel("Nearest-neighbour Chamfer distance to training set")
    ax.set_ylabel("Count")
    ax.set_title("Memorization Check: are generated shapes copies of training data?\n"
                 "(generated overlapping or right of baseline = NOT memorizing)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out/"memorization_check.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Summary bar for fidelity/coverage/generalization ----
    print(f"\n{'='*60}")
    print("GENERATIVE EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  FIDELITY    MMD   = {mmd:.4f}  (lower is better)")
    print(f"  COVERAGE    COV   = {cov*100:.1f}%  (higher is better)")
    print(f"  JOINT       1-NNA = {nna*100:.1f}%  (50% is ideal)")
    print(f"  GENERALIZE  mem-ratio = {mem_ratio:.2f}  (~1 healthy, <<1 = copying)")
    print(f"\nSaved: generative_eval.json, memorization_check.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=Path, required=True)
    p.add_argument("--vae_ckpt",      type=Path, required=True)
    p.add_argument("--out_dir",       type=Path, required=True)
    p.add_argument("--n_gen",         type=int, default=100)
    p.add_argument("--n_ref",         type=int, default=100)
    p.add_argument("--resolution",    type=int, default=96)
    p.add_argument("--offset",        type=float, default=0.01)
    args = p.parse_args()
    run(args)