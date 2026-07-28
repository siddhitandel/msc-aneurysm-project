# Implicit Generative Modelling of 3D Brain Aneurysm Geometries

**MSc Advanced Computer Science (AI) dissertation project — University of Leeds**

A deep generative model that learns the 3D shape space of intracranial
(brain) aneurysms, for shape analysis, synthetic shape generation, and a study
of whether generated shapes can augment rupture-risk prediction.

---

## Overview

Brain aneurysms rarely but catastrophically rupture, and their 3D shape is
believed to carry predictive signal. Labelled data, however, is scarce and
imbalanced. This project builds a variational autoencoder over a
signed-distance-function (SDF) representation of aneurysm surfaces, then uses
it to:

1. **Reconstruct and analyse** aneurysm geometry,
2. **Generate** realistic synthetic aneurysms, and
3. **Test data augmentation** — whether synthetic ruptured aneurysms improve a
   rupture-risk classifier.

The headline outcome is a coherent, rigorously-evaluated negative result:
the model learns *geometry* rather than *rupture status*, and learns the
*common* shape modes rather than the *rare* ones — a single structural
limitation that explains the model's volume bias, its limited generative
coverage, and why synthetic augmentation does not improve rupture prediction.

---

## Key results

| Result | Finding |
|---|---|
| Reconstruction | VAE validation SDF loss **0.017**; 735 shapes, 0 errors |
| Smoothness–fidelity tradeoff | Plain AE reconstructs best (0.0135) but is **0% samplable**; only the full VAE generates (77–90%) |
| Volume bias | Raw generated volume **×0.28** of real; a calibrated **+0.01 level-set offset** corrects it to **0.85 ± 0.07** (5-seed, indistinguishable from real, KS p>0.05) |
| Deep vs classical rupture prediction | Deep latent **0.739 ± 0.015** AUC ≈ classical morphometrics **0.748** (within noise, 10 seeds) |
| Generative evaluation | Fidelity (MMD) 0.204 good; **Coverage ~30% (limited)**; memorization ratio **0.94, 0/100 copies (not memorizing)** |
| Coverage is structural | Neither higher sampling temperature nor location-conditioning meaningfully improves coverage (+3.6pp weighted) |
| Augmentation | Synthetic ruptured shapes **do not** improve prediction (AUC −0.05) — explained by the above |

---

## Method

```
  Mesh (.vtp)                                    New shape
      │                                              ▲
      ▼                                              │
  Preprocess ──► Point cloud + SDF samples          │
                        │                            │
                        ▼                            │
              PointNet encoder                 Marching cubes
                        │                            ▲
                        ▼                            │
             512-d variational latent ──► SDF decoder (+ positional encoding)
                        │                            ▲
                        └──────── sample z ~ N(0,I) ─┘
```

- **Representation:** signed distance function (SDF) sampled near-surface (two
  noise scales) + uniform; surface point clouds for the encoder.
- **Model:** PointNet-style encoder (LayerNorm, no BatchNorm) → 512-d
  variational latent; 8-layer implicit SDF decoder with a skip connection and
  NeRF-style positional encoding; raw linear output. ~3.35M parameters.
- **Training:** clamped-L1 SDF loss + KL with free bits (0.5) and annealing
  (~150 epochs); Adam, cosine LR, gradient clipping; 800 epochs.
- **Generation:** sample a latent code, decode the SDF field, extract the mesh
  at a calibrated +0.01 level-set offset.

---

## Repository structure

```
aneurysm_project/
├── scripts/                        # all pipeline and experiment code
│   ├── augmentation_experiment.py  # Stage 2: synthetic augmentation study
│   ├── coverage_temperature_sweep.py  # coverage vs sampling temperature
│   ├── cvae_model.py               # rupture-conditioned VAE
│   ├── fair_coverage_comparison.py # controlled conditional-vs-unconditional coverage
│   ├── generative_eval.py          # fidelity/coverage/generalization + memorization
│   ├── level_set_check.py          # SDF level-set diagnostic + offset sweep
│   ├── location_cvae.py            # location(territory)-conditioned CVAE
│   ├── mesh_export_and_compare.py  # clean mesh export + real-vs-generated difference
│   ├── morphometric_baseline.py    # classical LogReg/RF on 12 clinical indices
│   ├── multiseed_classifier.py     # multi-seed CIs for the classifier comparison
│   ├── offset_robustness.py        # multi-seed validation of the volume-bias fix
│   └── train_cvae.py               # CVAE training
├── jobs/                           # SLURM job scripts (HPC)
├── results*/                       # result summaries (JSON) and figures (PNG)
└── models/                         # trained checkpoints (best_*.pt)
```

> **Note on large files.** The raw dataset, processed SDF samples, and the
> Stage-1 reconstruction checkpoints are excluded from version control (see
> `.gitignore`). Only the final model checkpoints, result summaries, and
> figures are tracked. Everything else is regenerable from the scripts.

---

## Data

**AneuX** intracranial aneurysm database
(Zenodo, DOI [10.5281/zenodo.6678442](https://doi.org/10.5281/zenodo.6678442)).
750 aneurysms; 735 with rupture labels (474 unruptured / 261 ruptured); the
"dome" cut is used. Split 70/15/15 (seed 42) → 514 train / 110 val / 111 test,
held fixed across every model and classifier. The dataset is **not**
redistributed here — download it from Zenodo and point the scripts at it.

---

## Reproducing the results

Environment (conda):

```bash
conda env create -f aneuseg_env.yml
conda activate aneuseg
```

Representative commands (paths assume the AneuX data is available locally):

```bash
# Generative evaluation — fidelity / coverage / generalization + memorization
python scripts/generative_eval.py \
    --processed_dir data/processed_sdf \
    --vae_ckpt      models/vae_compare/best.pt \
    --out_dir       results_genereval --n_gen 100 --offset 0.01

# Volume-bias correction, validated across seeds
python scripts/offset_robustness.py \
    --processed_dir data/processed_sdf \
    --vae_ckpt      models/vae_compare/best.pt \
    --out_dir       results_robustness --offset 0.01 --n_seeds 5

# Multi-seed deep-vs-classical rupture classifier comparison
python scripts/multiseed_classifier.py \
    --baseline_dir  results_classifier \
    --morpho_csv    <path>/clinical.csv \
    --morpho_feat   <path>/morpho-per-cut.csv \
    --out_dir       results_multiseed --n_seeds 10

# Location-conditioned generation (train, then evaluate per-territory coverage)
python scripts/location_cvae.py train    --processed_dir data/processed_sdf \
    --clinical_csv <path>/clinical.csv --out_dir models/location_cvae
python scripts/location_cvae.py evaluate --processed_dir data/processed_sdf \
    --clinical_csv <path>/clinical.csv \
    --cvae_ckpt models/location_cvae/best_loc_cvae.pt --out_dir results_location
```

Most training was run on the University of Leeds AIRE HPC cluster via the
SLURM scripts in `jobs/`.

---

## Notes & limitations

- Small test set (111 shapes, 37 ruptured) → wide confidence intervals; the
  deep-vs-classical AUC gap is within noise.
- The +0.01 level-set offset is **calibrated to this model and dataset**, not
  a universal constant.
- Trained on the dome cut only.
- Generative coverage is a **characterised limitation**, not resolved —
  reported honestly with controlled experiments (temperature sweep,
  location-conditioning) showing it is structural.

---

## Acknowledgements

AneuX database and its contributing cohorts (hug2016, AneuRIST, AneuRisk).
Computation on the University of Leeds AIRE HPC cluster.
