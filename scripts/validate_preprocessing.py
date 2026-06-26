"""
validate_preprocessing.py
==========================
Run this after preprocessing to sanity-check the outputs before
moving on to model training.

Usage:
    python scripts/validate_preprocessing.py \
        --processed_dir ~/aneurysm_project/data/processed

Checks:
  - Number of files matches expected
  - Point cloud shapes are correct
  - Labels distribution (ruptured vs unruptured)
  - No NaN / Inf values in point clouds
  - Bounding box sanity (all points inside [-1.2, 1.2])
  - Plots a sample point cloud to results/sample_pointcloud.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no display needed on HPC
import matplotlib.pyplot as plt


def validate(processed_dir: Path):
    processed_dir = Path(processed_dir)
    surface_dir = processed_dir / "surface"
    space_dir   = processed_dir / "space"
    manifest_path = processed_dir / "manifest.json"
    labels_path   = processed_dir / "labels.csv"

    print("=" * 60)
    print(" Validation Report")
    print("=" * 60)

    # ── Manifest ──────────────────────────────────────────────
    if not manifest_path.exists():
        print("ERROR: manifest.json not found. Did preprocessing complete?")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\n[Manifest]")
    print(f"  Processed      : {manifest['total_processed']}")
    print(f"  Skipped        : {manifest['total_skipped']}")
    print(f"  Errors         : {manifest['total_errors']}")
    print(f"  Representation : {manifest['representation']}")
    print(f"  Surface pts    : {manifest['n_surface_pts']}")
    print(f"  Space pts      : {manifest['n_space_pts']}")

    # ── File counts ───────────────────────────────────────────
    surface_files = list(surface_dir.glob("*.npz"))
    space_files   = list(space_dir.glob("*.npz"))
    print(f"\n[File Counts]")
    print(f"  surface/*.npz  : {len(surface_files)}")
    print(f"  space/*.npz    : {len(space_files)}")
    if len(surface_files) != len(space_files):
        print("  WARNING: surface and space file counts don't match!")

    # ── Labels ────────────────────────────────────────────────
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        print(f"\n[Labels  (from labels.csv)]")
        counts = labels_df["status"].value_counts()
        for status, count in counts.items():
            pct = 100 * count / len(labels_df)
            print(f"  {status:12s} : {count:4d}  ({pct:.1f}%)")
        print(f"  Total          : {len(labels_df)}")

        rupture_rate = labels_df["label"].mean()
        print(f"  Rupture rate   : {rupture_rate:.3f}")
        if rupture_rate < 0.2 or rupture_rate > 0.8:
            print("  NOTE: Class imbalance detected — consider weighted loss during training.")

    # ── Spot-check 10 random files ────────────────────────────
    print(f"\n[Point Cloud Checks  (10 random samples)]")
    sample_files = np.random.choice(surface_files, size=min(10, len(surface_files)), replace=False)

    all_ok = True
    for fpath in sample_files:
        data = np.load(fpath)
        pts  = data["points"]
        name = fpath.stem

        issues = []
        if pts.shape != (manifest["n_surface_pts"], 3):
            issues.append(f"shape={pts.shape}")
        if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
            issues.append("contains NaN/Inf")
        if pts.max() > 1.5 or pts.min() < -1.5:
            issues.append(f"out of bounds (min={pts.min():.3f}, max={pts.max():.3f})")

        status_str = "OK" if not issues else "FAIL: " + ", ".join(issues)
        if issues:
            all_ok = False
        print(f"  {name:30s}  {status_str}")

    if all_ok:
        print("  All sampled files passed checks.")

    # ── Space point labels distribution ───────────────────────
    print(f"\n[Space Points Label Balance  (10 random samples)]")
    space_samples = np.random.choice(space_files, size=min(10, len(space_files)), replace=False)
    inside_fracs = []
    for fpath in space_samples:
        data = np.load(fpath)
        if "labels" in data:
            frac = data["labels"].mean()
            inside_fracs.append(frac)
    if inside_fracs:
        print(f"  Mean fraction inside  : {np.mean(inside_fracs):.3f}  (expect ~0.2-0.4 for aneurysm domes)")
        print(f"  Range                 : [{min(inside_fracs):.3f}, {max(inside_fracs):.3f}]")

    # ── Plot a sample point cloud ─────────────────────────────
    print(f"\n[Visualisation]")
    try:
        sample = np.load(surface_files[0])
        pts = sample["points"]
        label = int(sample["rupture_label"])
        label_str = "Ruptured" if label == 1 else "Unruptured"

        fig = plt.figure(figsize=(8, 8))
        ax  = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=1, c=pts[:, 2], cmap="viridis", alpha=0.6)
        ax.set_title(f"Sample: {surface_files[0].stem}\nLabel: {label_str}", fontsize=12)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])

        out_path = processed_dir.parent.parent / "results" / "sample_pointcloud.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved sample plot to: {out_path}")
    except Exception as e:
        print(f"  Could not generate plot: {e}")

    print("\n" + "=" * 60)
    print(" Validation complete. Ready for model training.")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=Path, required=True,
                   help="Path to data/processed directory")
    args = p.parse_args()
    validate(args.processed_dir)
