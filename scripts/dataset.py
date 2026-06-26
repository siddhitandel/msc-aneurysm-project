"""
dataset.py
==========
PyTorch Dataset classes for loading preprocessed AneuX point clouds.

Two datasets:
  OccupancyDataset  — loads space points + occupancy labels
                      used to train the implicit reconstruction MLP
  SurfaceDataset    — loads surface point clouds
                      used to train the autoencoder encoder later

Both support train/val/test splits via a split argument and
a fixed random seed for reproducibility.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Occupancy Dataset  (reconstruction model training)
# ─────────────────────────────────────────────────────────────────────────────

class OccupancyDataset(Dataset):
    """
    Loads preprocessed space/*.npz files for occupancy network training.

    Each item returns:
      points : (N, 3)  float32  3D query coordinates, normalised to unit sphere
      labels : (N,)    float32  1.0 = inside, 0.0 = outside
      idx    : int     index into self.ids (for debugging)

    Args:
      processed_dir : path to data/processed/
      split         : 'train', 'val', or 'test'
      train_frac    : fraction of data for training   (default 0.70)
      val_frac      : fraction of data for validation (default 0.15)
                      remainder goes to test
      seed          : random seed for reproducible splits (default 42)
      n_points      : subsample this many points per shape per epoch
                      None = use all points in the file
    """

    def __init__(self,
                 processed_dir: str | Path,
                 split:         str   = "train",
                 train_frac:    float = 0.70,
                 val_frac:      float = 0.15,
                 seed:          int   = 42,
                 n_points:      int   = None):

        self.space_dir  = Path(processed_dir) / "space"
        self.split      = split
        self.n_points   = n_points

        # Load manifest to get the list of processed IDs
        manifest_path = Path(processed_dir) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found in {processed_dir}. "
                "Run preprocess.py first."
            )
        with open(manifest_path) as f:
            manifest = json.load(f)

        all_ids = manifest["aneurysm_ids"]

        # Load labels
        labels_df = pd.read_csv(Path(processed_dir) / "labels.csv")
        self.label_map = dict(zip(
            labels_df["dataset"].astype(str),
            labels_df["label"].astype(int)
        ))

        # Reproducible split
        rng = np.random.default_rng(seed)
        indices = np.arange(len(all_ids))
        rng.shuffle(indices)

        n_train = int(len(all_ids) * train_frac)
        n_val   = int(len(all_ids) * val_frac)

        if split == "train":
            split_indices = indices[:n_train]
        elif split == "val":
            split_indices = indices[n_train : n_train + n_val]
        elif split == "test":
            split_indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test'. Got: {split}")

        self.ids = [all_ids[i] for i in split_indices]

        print(f"OccupancyDataset [{split}]: {len(self.ids)} shapes  "
              f"({sum(self.label_map.get(i, 0) for i in self.ids)} ruptured)")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        aneurysm_id = self.ids[idx]
        data = np.load(self.space_dir / f"{aneurysm_id}.npz")

        points = data["points"]   # (N, 3)
        labels = data["labels"]   # (N,)

        # Optional: subsample points for memory efficiency / data augmentation
        if self.n_points is not None and len(points) > self.n_points:
            chosen = np.random.choice(len(points), self.n_points, replace=False)
            points = points[chosen]
            labels = labels[chosen]

        return {
            "points"        : torch.from_numpy(points),          # (N, 3)
            "labels"        : torch.from_numpy(labels),          # (N,)
            "rupture_label" : torch.tensor(self.label_map.get(aneurysm_id, -1),
                                           dtype=torch.long),
            "id"            : aneurysm_id,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Surface Dataset  (autoencoder encoder input — used later)
# ─────────────────────────────────────────────────────────────────────────────

class SurfaceDataset(Dataset):
    """
    Loads preprocessed surface/*.npz files.
    Used for the generative autoencoder encoder input.

    Each item returns:
      points : (N, 3)  float32  surface point cloud, normalised to unit sphere
    """

    def __init__(self,
                 processed_dir: str | Path,
                 split:         str   = "train",
                 train_frac:    float = 0.70,
                 val_frac:      float = 0.15,
                 seed:          int   = 42,
                 n_points:      int   = 4096,
                 augment:       bool  = False):

        self.surface_dir = Path(processed_dir) / "surface"
        self.split       = split
        self.n_points    = n_points
        self.augment     = augment

        manifest_path = Path(processed_dir) / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        all_ids = manifest["aneurysm_ids"]

        labels_df = pd.read_csv(Path(processed_dir) / "labels.csv")
        self.label_map = dict(zip(
            labels_df["dataset"].astype(str),
            labels_df["label"].astype(int)
        ))

        rng = np.random.default_rng(seed)
        indices = np.arange(len(all_ids))
        rng.shuffle(indices)

        n_train = int(len(all_ids) * train_frac)
        n_val   = int(len(all_ids) * val_frac)

        if split == "train":
            split_indices = indices[:n_train]
        elif split == "val":
            split_indices = indices[n_train : n_train + n_val]
        elif split == "test":
            split_indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test'. Got: {split}")

        self.ids = [all_ids[i] for i in split_indices]

        print(f"SurfaceDataset [{split}]: {len(self.ids)} shapes")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        aneurysm_id = self.ids[idx]
        data   = np.load(self.surface_dir / f"{aneurysm_id}.npz")
        points = data["points"].copy()   # (N, 3)

        # Subsample / pad to fixed size
        if len(points) >= self.n_points:
            chosen = np.random.choice(len(points), self.n_points, replace=False)
            points = points[chosen]
        else:
            # Pad by repeating random points (rare edge case)
            extra  = np.random.choice(len(points),
                                      self.n_points - len(points), replace=True)
            points = np.concatenate([points, points[extra]], axis=0)

        # Optional augmentation: random rotation around Z axis
        if self.augment and self.split == "train":
            angle  = np.random.uniform(0, 2 * np.pi)
            c, s   = np.cos(angle), np.sin(angle)
            R      = np.array([[c, -s, 0],
                                [s,  c, 0],
                                [0,  0, 1]], dtype=np.float32)
            points = points @ R.T

        return {
            "points"        : torch.from_numpy(points),
            "rupture_label" : torch.tensor(self.label_map.get(aneurysm_id, -1),
                                           dtype=torch.long),
            "id"            : aneurysm_id,
        }