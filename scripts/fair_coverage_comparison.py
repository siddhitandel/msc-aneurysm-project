"""
fair_coverage_comparison.py
===========================
Controlled comparison: does location-conditioning actually improve coverage?

The naive comparison (per-territory conditional COV vs the global
unconditional COV) is confounded:
  - different numbers of generated shapes (40 vs 100)
  - different numbers of real reference shapes per territory (9 to 54)
  - COV is biased upward when n_real is small relative to n_gen
    (with only 9 real shapes, even near-random generation covers most of them)
  - a territory with more real shapes than generated shapes is CAPPED
    (40 generated cannot cover more than 40 of ICA's 54 real shapes)

This script removes those confounds. For EACH territory it scores, on the
SAME real reference shapes and with the SAME number of generated shapes:
    A) shapes from the UNCONDITIONAL VAE
    B) shapes from the LOCATION-CONDITIONED CVAE (conditioned on that territory)
so the only difference is the conditioning.

It also reports:
  - COV_max     : the ceiling given n_gen and n_real (min(n_gen,n_real)/n_real)
  - COV_random  : expected COV if each generated shape picked a nearest real
                  uniformly at random = (1 - (1 - 1/n_real)^n_gen)
                  -> a null baseline; being below it means generated shapes
                     cluster rather than spread
  - COV_norm    : achieved COV divided by COV_random (1.0 = as spread out as
                  random assignment, <1 = clustered)

Usage:
  python scripts/fair_coverage_comparison.py \
      --processed_dir aneurysm_project/data/processed_sdf \
      --clinical_csv  ~/MSc_Project/6678442/data/clinical.csv \
      --vae_ckpt      aneurysm_project/models/vae_compare/best.pt \
      --loc_ckpt      aneurysm_project/models/location_cvae/best_loc_cvae.pt \
      --out_dir       aneurysm_project/results_fair_coverage \
      --n_gen 40
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
from location_cvae import LocCVAE, load_location_map, TERRITORIES
from dataset import SurfaceDataset


# ---- shared geometry helpers ------------------------------------------------

def _grid_pts(res):
    lin = np.linspace(-1, 1, res)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], -1).astype(np.float32)


def sdf_grid_uncond(model, z, res, device, batch=65536):
    pts = torch.from_numpy(_grid_pts(res)).to(device)
    vals = []
    with torch.no_grad():
        for i in range(0, len(pts), batch):
            c = pts[i:i+batch].unsqueeze(0)
            vals.append(model.decode(c, z if z.dim()==2 else z.unsqueeze(0)).squeeze().cpu().numpy())
    return np.concatenate(vals).reshape(res, res, res)


def sdf_grid_cond(model, z, lab, res, device, batch=65536):
    pts = torch.from_numpy(_grid_pts(res)).to(device)
    vals = []
    with torch.no_grad():
        for i in range(0, len(pts), batch):
            c = pts[i:i+batch].unsqueeze(0)
            vals.append(model.decode(c, z if z.dim()==2 else z.unsqueeze(0), lab).squeeze().cpu().numpy())
    return np.concatenate(vals).reshape(res, res, res)


def grid_to_points(g, level, n=2048):
    try:
        v, f, _, _ = measure.marching_cubes(g, level=level)
    except (ValueError, RuntimeError):
        return None
    r = g.shape[0]; v = v/(r-1)*2 - 1
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    if len(m.faces) == 0: return None
    p, _ = trimesh.sample.sample_surface(m, n)
    return p.astype(np.float32)


def chamfer(a, b):
    ta, tb = cKDTree(a), cKDTree(b)
    return float(tb.query(a)[0].mean() + ta.query(b)[0].mean())


def coverage(gen, real):
    """COV plus the diagnostic ceilings/nulls."""
    D = np.zeros((len(gen), len(real)))
    for i, g in enumerate(gen):
        for j, r in enumerate(real):
            D[i, j] = chamfer(g, r)
    cov = len(set(D.argmin(axis=1).tolist())) / len(real)
    mmd = float(D.min(axis=0).mean())
    n_g, n_r = len(gen), len(real)
    cov_max = min(n_g, n_r) / n_r
    cov_rand = 1.0 - (1.0 - 1.0/n_r)**n_g
    return {
        "COV": cov, "MMD": mmd,
        "COV_max": cov_max, "COV_random": cov_rand,
        "COV_norm": cov / cov_rand if cov_rand > 0 else float("nan"),
        "n_gen": n_g, "n_real": n_r,
    }


def run(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    res = args.resolution

    # --- load both models ---
    ck = torch.load(args.vae_ckpt, map_location=dev)
    uncond = AneurysmAE(latent_dim=ck["latent_dim"], hidden_dim=ck["hidden_dim"],
                        n_freqs=ck["n_freqs"], variational=ck["variational"],
                        beta=ck.get("beta",1.0), mode_name=ck["mode"]).to(dev)
    uncond.load_state_dict(ck["model_state_dict"]); uncond.eval()
    print(f"Unconditional VAE: val recon {ck['val_recon']:.4f}")

    lck = torch.load(args.loc_ckpt, map_location=dev)
    cond = LocCVAE(n_freqs=lck["n_freqs"]).to(dev)
    cond.load_state_dict(lck["model_state_dict"]); cond.eval()
    print(f"Location CVAE:     val recon {lck['val_recon']:.4f}")

    # --- real test shapes by territory ---
    loc = load_location_map(args.clinical_csv)
    ds = SurfaceDataset(args.processed_dir, split="test", n_points=2048)
    real_by = {t: [] for t in range(len(TERRITORIES))}
    for sid in ds.ids:
        if sid in loc:
            p = np.load(Path(args.processed_dir)/"surface"/f"{sid}.npz")["points"]
            if len(p) > 2048:
                p = p[np.random.choice(len(p), 2048, replace=False)]
            real_by[loc[sid]].append(p.astype(np.float32))

    # --- generate ONE pool of unconditional shapes (reused for every territory) ---
    print(f"\nGenerating {args.n_gen} unconditional shapes...")
    uncond_gen = []
    tries = 0
    while len(uncond_gen) < args.n_gen and tries < args.n_gen*4:
        tries += 1
        z = torch.randn(1, uncond.latent_dim, device=dev)
        p = grid_to_points(sdf_grid_uncond(uncond, z, res, dev), args.offset)
        if p is not None: uncond_gen.append(p)
    print(f"  got {len(uncond_gen)}")

    results = {}
    print(f"\n{'='*74}")
    print("FAIR PER-TERRITORY COMPARISON  (same real shapes, same n_gen)")
    print(f"{'='*74}")
    hdr = f"{'Territory':10s} {'n_real':>6} {'COVunc':>8} {'COVcond':>8} {'delta':>7} {'ceiling':>8} {'random':>8}"
    print(hdr); print("  " + "-"*70)

    for t, tname in enumerate(TERRITORIES):
        reals = real_by[t]
        if len(reals) < 3:
            print(f"{tname:10s} only {len(reals)} real shapes - skipped")
            continue

        # conditional generation for THIS territory
        cgen = []; tries = 0
        while len(cgen) < args.n_gen and tries < args.n_gen*4:
            tries += 1
            lab = torch.tensor([t], dtype=torch.long, device=dev)
            z = torch.randn(1, cond.latent_dim, device=dev)
            p = grid_to_points(sdf_grid_cond(cond, z, lab, res, dev), args.offset)
            if p is not None: cgen.append(p)

        # score BOTH against the same real set
        cu = coverage(uncond_gen, reals)
        cc = coverage(cgen, reals)
        delta = (cc["COV"] - cu["COV"]) * 100
        results[tname] = {"unconditional": cu, "conditional": cc,
                          "delta_COV_pct": delta}
        print(f"{tname:10s} {len(reals):>6} {cu['COV']*100:>7.1f}% {cc['COV']*100:>7.1f}% "
              f"{delta:>+6.1f}pp {cc['COV_max']*100:>7.1f}% {cc['COV_random']*100:>7.1f}%")

    # weighted (by n_real) mean delta = the honest headline
    tot_real = sum(results[t]["conditional"]["n_real"] for t in results)
    w_unc = sum(results[t]["unconditional"]["COV"]*results[t]["conditional"]["n_real"] for t in results)/tot_real
    w_con = sum(results[t]["conditional"]["COV"]*results[t]["conditional"]["n_real"] for t in results)/tot_real
    simple_unc = np.mean([results[t]["unconditional"]["COV"] for t in results])
    simple_con = np.mean([results[t]["conditional"]["COV"] for t in results])

    results["_summary"] = {
        "weighted_COV_unconditional_pct": w_unc*100,
        "weighted_COV_conditional_pct":   w_con*100,
        "weighted_delta_pp":              (w_con-w_unc)*100,
        "unweighted_COV_unconditional_pct": simple_unc*100,
        "unweighted_COV_conditional_pct":   simple_con*100,
        "note": ("Weighted means are weighted by n_real per territory and are the "
                 "honest headline; unweighted means over-count small, noisy territories."),
    }
    json.dump(results, open(out/"fair_coverage.json","w"), indent=2)

    print("\n" + "="*74)
    print(f"  Sample-weighted COV : unconditional {w_unc*100:.1f}%  ->  conditional {w_con*100:.1f}%"
          f"   ({(w_con-w_unc)*100:+.1f} pp)")
    print(f"  Unweighted mean COV : unconditional {simple_unc*100:.1f}%  ->  conditional {simple_con*100:.1f}%")
    print("  (weighted is the fairer number; unweighted over-counts tiny territories)")

    # --- paired bar chart ---
    terrs = [t for t in TERRITORIES if t in results]
    x = np.arange(len(terrs)); w = 0.36
    cu = [results[t]["unconditional"]["COV"]*100 for t in terrs]
    cc = [results[t]["conditional"]["COV"]*100 for t in terrs]
    ceil = [results[t]["conditional"]["COV_max"]*100 for t in terrs]
    rnd  = [results[t]["conditional"]["COV_random"]*100 for t in terrs]

    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(x-w/2, cu, w, label="Unconditional VAE", color="#888888")
    ax.bar(x+w/2, cc, w, label="Location-conditioned CVAE", color="#1C7293")
    ax.plot(x, ceil, "_", markersize=26, color="#C0392B", label="ceiling (n_gen limit)")
    ax.plot(x, rnd,  "_", markersize=26, color="#D98A00", label="random-assignment null")
    ax.set_xticks(x); ax.set_xticklabels([f"{t}\n(n={results[t]['conditional']['n_real']})" for t in terrs])
    ax.set_ylabel("Coverage (COV) %")
    ax.set_title("Coverage: conditional vs unconditional, scored on identical real shapes")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    for i,(a,b) in enumerate(zip(cu,cc)):
        ax.text(i-w/2, a+1, f"{a:.0f}", ha="center", fontsize=9)
        ax.text(i+w/2, b+1, f"{b:.0f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out/"fair_coverage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: fair_coverage.json, fair_coverage.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=Path, required=True)
    p.add_argument("--clinical_csv",  type=Path, required=True)
    p.add_argument("--vae_ckpt",      type=Path, required=True)
    p.add_argument("--loc_ckpt",      type=Path, required=True)
    p.add_argument("--out_dir",       type=Path, required=True)
    p.add_argument("--n_gen",         type=int, default=40)
    p.add_argument("--resolution",    type=int, default=96)
    p.add_argument("--offset",        type=float, default=0.01)
    args = p.parse_args()
    run(args)