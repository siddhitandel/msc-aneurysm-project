"""
preprocess.py
=============
Converts AneuX aneurysm surface meshes (VTP format) into
normalised point clouds ready for implicit neural network training.

Actual AneuX folder structure (confirmed):
  data/clinical.csv
  models/aneurysms/remeshed/area-005/<dataset_id>_<cut>.vtp
    e.g.  ANSYS_UNIGE_09_dome.vtp
          ANSYS_UNIGE_09_cut1.vtp
          ANSYS_UNIGE_09_ninja.vtp

What this script does:
  1. Reads clinical.csv  ->  rupture status label per aneurysm
  2. For each aneurysm, finds its dome cut VTP file in area-005/
  3. Normalises the mesh  ->  centred at origin, scaled to unit sphere
  4. Samples a fixed-size point cloud from the mesh surface
  5. Also samples off-surface points (needed for occupancy / SDF training)
  6. Saves everything as .npz files in data/processed/

Usage (interactive test on a few shapes):
  conda activate aneuseg
  python scripts/preprocess.py \
      --data_root ~/6678442 \
      --out_dir   ~/aneurysm_project/data/processed \
      --n_surface 4096 \
      --n_space   4096 \
      --limit     5

Usage (full SLURM job):
  sbatch jobs/preprocess.job
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import trimesh
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Mesh I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_vtp_as_trimesh(vtp_path: Path) -> trimesh.Trimesh:
    """
    Read a VTP surface mesh with PyVista and convert to trimesh.
    """
    mesh_pv  = pv.read(str(vtp_path))
    surface  = mesh_pv.extract_surface().triangulate()
    vertices = np.array(surface.points, dtype=np.float32)
    # PyVista faces: [n_verts, v0, v1, v2, n_verts, v0, v1, v2, ...]
    faces = surface.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_mesh(mesh: trimesh.Trimesh):
    """
    Centre at origin, scale to unit sphere.
    Returns (mesh, centroid, scale) so the transform can be inverted later.

    IMPORTANT: We build a BRAND NEW Trimesh from the normalised vertices.
    Editing mesh.vertices in place does NOT invalidate trimesh's cached
    triangle tree / BVH used by proximity.closest_point, which caused SDF
    distances to be computed against the original (unnormalised) geometry.
    Constructing a fresh mesh guarantees all cached structures are rebuilt.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    centroid = verts.mean(axis=0)
    verts = verts - centroid
    scale = float(np.linalg.norm(verts, axis=1).max())
    verts = verts / scale

    # Fresh mesh — no stale cache
    norm_mesh = trimesh.Trimesh(vertices=verts.astype(np.float32),
                                faces=np.asarray(mesh.faces),
                                process=False)
    return norm_mesh, centroid, scale


# ─────────────────────────────────────────────────────────────────────────────
# Point cloud sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_surface_points(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
    """Sample n points uniformly from the mesh surface. Returns (n, 3) float32."""
    points, _ = trimesh.sample.sample_surface(mesh, n)
    return points.astype(np.float32)


def sample_occupancy_points(mesh: trimesh.Trimesh,
                            n_near: int,
                            n_far: int,
                            noise_std: float = 0.02):
    """
    Sample space points with binary occupancy labels.
      n_near : perturbed surface points  ->  fine boundary detail
      n_far  : uniform random in [-1.2, 1.2]^3  ->  global context

    Returns:
      points : (n_near + n_far, 3)  float32
      labels : (n_near + n_far,)    float32   1.0=inside  0.0=outside
    """
    surf_pts, _ = trimesh.sample.sample_surface(mesh, n_near)
    noise    = np.random.randn(*surf_pts.shape).astype(np.float32) * noise_std
    near_pts = (surf_pts + noise).astype(np.float32)
    far_pts  = np.random.uniform(-1.2, 1.2, size=(n_far, 3)).astype(np.float32)
    all_pts  = np.concatenate([near_pts, far_pts], axis=0)
    labels   = mesh.ray.contains_points(all_pts).astype(np.float32)
    return all_pts, labels


def sample_sdf_points(mesh: trimesh.Trimesh,
                      n_near: int,
                      n_far: int,
                      noise_std: float = 0.02):
    """
    Sample space points with signed distance values.
    Negative = inside, positive = outside, zero = on surface.

    NOTE: We compute distances using a KD-tree to a dense set of surface
    samples rather than trimesh.proximity.closest_point, which returns
    incorrect (wildly inflated) distances in this trimesh version even on
    correctly-normalised meshes. The KD-tree gives accurate unsigned
    distance; the sign comes from mesh.contains (inside/outside test, which
    is reliable). Distances are in normalised units (~[0, 2]).

    Near-surface points are sampled at two noise scales for dense
    supervision around the surface. Far points give global context.

    Returns:
      points : (n_near + n_far, 3)  float32
      sdfs   : (n_near + n_far,)    float32
    """
    from scipy.spatial import cKDTree

    # Dense surface sample for the KD-tree (defines the surface)
    dense_surf, _ = trimesh.sample.sample_surface(mesh, 50000)
    tree = cKDTree(np.asarray(dense_surf, dtype=np.float32))

    # Query points: two near-surface noise scales + far field
    n_tight  = n_near // 2
    n_medium = n_near - n_tight

    surf_tight,  _ = trimesh.sample.sample_surface(mesh, n_tight)
    surf_medium, _ = trimesh.sample.sample_surface(mesh, n_medium)
    near_tight  = surf_tight  + np.random.randn(*surf_tight.shape ).astype(np.float32) * 0.01
    near_medium = surf_medium + np.random.randn(*surf_medium.shape).astype(np.float32) * 0.05
    far_pts     = np.random.uniform(-1.1, 1.1, size=(n_far, 3)).astype(np.float32)

    all_pts = np.concatenate([near_tight, near_medium, far_pts],
                             axis=0).astype(np.float32)

    # Unsigned distance via KD-tree (reliable)
    unsigned_dist, _ = tree.query(all_pts)
    sd = unsigned_dist.astype(np.float32)

    # Sign via inside/outside test (scale-invariant, reliable)
    inside = mesh.contains(all_pts)
    sd[inside] *= -1.0

    return all_pts, sd


# ─────────────────────────────────────────────────────────────────────────────
# Mesh lookup  (flat dir: area-005/<id>_dome.vtp)
# ─────────────────────────────────────────────────────────────────────────────

def find_mesh_file(aneurysm_id: str, area005_dir: Path):
    """
    Find the dome-cut VTP file for a given dataset ID.
    Confirmed AneuX layout: all VTPs flat inside area-005/
    Named:  <dataset_id>_<cut>.vtp
    e.g.    ANSYS_UNIGE_09_dome.vtp

    Priority: dome > ninja > cut1 > cut2
    """
    for cut in ["dome", "ninja", "cut1", "cut2"]:
        candidate = area005_dir / f"{aneurysm_id}_{cut}.vtp"
        if candidate.exists():
            return candidate
    # Last resort: any file starting with this ID
    matches = list(area005_dir.glob(f"{aneurysm_id}_*.vtp"))
    return matches[0] if matches else None


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(data_root: Path,
                    out_dir: Path,
                    n_surface: int       = 4096,
                    n_space: int         = 4096,
                    representation: str  = "occupancy",
                    min_faces: int       = 100,
                    limit: int           = None):

    # Confirmed exact paths from directory inspection
    clinical_path = data_root / "data" / "clinical.csv"
    area005_dir   = data_root / "models" / "aneurysms" / "remeshed" / "area-005"

    if not clinical_path.exists():
        raise FileNotFoundError(f"clinical.csv not found at: {clinical_path}")
    if not area005_dir.exists():
        raise FileNotFoundError(f"Mesh dir not found at: {area005_dir}")

    print(f"Clinical CSV  : {clinical_path}")
    print(f"Mesh dir      : {area005_dir}")
    print(f"Output dir    : {out_dir}")
    print(f"Representation: {representation}")
    print(f"Surface pts   : {n_surface}   Space pts: {n_space}\n")

    # Create output folders
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "surface").mkdir(exist_ok=True)
    (out_dir / "space").mkdir(exist_ok=True)

    # Load clinical labels
    clinical = pd.read_csv(clinical_path)
    print(f"Loaded clinical.csv : {len(clinical)} rows")
    print(f"  Rupture status available : {clinical['status'].notna().sum()} / {len(clinical)}")
    print(f"  Status counts:\n{clinical['status'].value_counts().to_string()}\n")

    clinical = clinical[clinical["status"].notna()].copy()
    clinical["label"] = (clinical["status"] == "ruptured").astype(int)

    if limit:
        clinical = clinical.head(limit)
        print(f"[--limit {limit}] Testing on first {limit} aneurysms only.\n")

    processed, skipped, errors = [], [], []

    for _, row in tqdm(clinical.iterrows(), total=len(clinical), desc="Preprocessing"):
        aneurysm_id = str(row["dataset"])
        label       = int(row["label"])
        source      = str(row["source"])
        location    = str(row.get("location", "unknown"))

        vtp_path = find_mesh_file(aneurysm_id, area005_dir)
        if vtp_path is None:
            tqdm.write(f"  SKIP {aneurysm_id}: no VTP file found")
            skipped.append(aneurysm_id)
            continue

        try:
            mesh = load_vtp_as_trimesh(vtp_path)
            if len(mesh.faces) < min_faces:
                tqdm.write(f"  SKIP {aneurysm_id}: only {len(mesh.faces)} faces")
                skipped.append(aneurysm_id)
                continue

            mesh, centroid, scale = normalise_mesh(mesh)
            surface_pts = sample_surface_points(mesh, n_surface)

            # Save surface point cloud (autoencoder encoder input)
            np.savez_compressed(
                out_dir / "surface" / f"{aneurysm_id}.npz",
                points        = surface_pts,
                centroid      = centroid,
                scale         = scale,
                rupture_label = label,
                source        = source,
                location      = location,
            )

            # Save space points + implicit labels
            if representation == "occupancy":
                space_pts, space_vals = sample_occupancy_points(
                    mesh, n_near=n_space // 2, n_far=n_space // 2
                )
                np.savez_compressed(
                    out_dir / "space" / f"{aneurysm_id}.npz",
                    points        = space_pts,
                    labels        = space_vals,
                    centroid      = centroid,
                    scale         = scale,
                    rupture_label = label,
                    source        = source,
                    location      = location,
                )
            else:
                space_pts, space_vals = sample_sdf_points(
                    mesh, n_near=n_space // 2, n_far=n_space // 2
                )
                np.savez_compressed(
                    out_dir / "space" / f"{aneurysm_id}.npz",
                    points        = space_pts,
                    sdfs          = space_vals,
                    centroid      = centroid,
                    scale         = scale,
                    rupture_label = label,
                    source        = source,
                    location      = location,
                )

            processed.append(aneurysm_id)

        except Exception as e:
            tqdm.write(f"  ERROR {aneurysm_id}: {e}")
            errors.append((aneurysm_id, str(e)))

    # Save manifest
    manifest = {
        "total_processed" : len(processed),
        "total_skipped"   : len(skipped),
        "total_errors"    : len(errors),
        "representation"  : representation,
        "n_surface_pts"   : n_surface,
        "n_space_pts"     : n_space,
        "aneurysm_ids"    : processed,
        "skipped_ids"     : skipped,
        "errors"          : errors,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Save clean labels CSV
    df_out = clinical[clinical["dataset"].astype(str).isin(processed)][
        ["dataset", "label", "status", "source", "location", "sex", "age"]
    ].copy()
    df_out.to_csv(out_dir / "labels.csv", index=False)

    print("\n" + "=" * 55)
    print(f"  Preprocessing complete")
    print(f"  Processed : {len(processed)}")
    print(f"  Skipped   : {len(skipped)}")
    print(f"  Errors    : {len(errors)}")
    print(f"  Output    : {out_dir}")
    print("=" * 55)
    if errors:
        print("\nFirst errors:")
        for eid, emsg in errors[:5]:
            print(f"  {eid}: {emsg}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="AneuX preprocessing pipeline")
    p.add_argument("--data_root",      type=Path, required=True,
                   help="Root of AneuX dataset (contains data/ and models/)")
    p.add_argument("--out_dir",        type=Path, required=True,
                   help="Output directory for processed .npz files")
    p.add_argument("--n_surface",      type=int,  default=4096,
                   help="Surface points per mesh (default: 4096)")
    p.add_argument("--n_space",        type=int,  default=4096,
                   help="Space points per mesh (default: 4096)")
    p.add_argument("--representation", type=str,  default="occupancy",
                   choices=["occupancy", "sdf"],
                   help="occupancy or sdf (default: occupancy)")
    p.add_argument("--min_faces",      type=int,  default=100,
                   help="Skip meshes with fewer faces (default: 100)")
    p.add_argument("--limit",          type=int,  default=None,
                   help="Only process first N shapes — for testing")
    args = p.parse_args()

    process_dataset(
        data_root      = args.data_root,
        out_dir        = args.out_dir,
        n_surface      = args.n_surface,
        n_space        = args.n_space,
        representation = args.representation,
        min_faces      = args.min_faces,
        limit          = args.limit,
    )