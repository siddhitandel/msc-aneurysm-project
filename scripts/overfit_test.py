"""
overfit_test.py
===============
Sanity check: can the VAE overfit a single shape?
If recon loss drops steadily, the architecture and training are correct.

Usage:
  python scripts/overfit_test.py
"""
import numpy as np
import torch
import sys
import glob

sys.path.insert(0, "aneurysm_project/scripts")
from vae_model import AneurysmVAE, vae_sdf_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AneurysmVAE(latent_dim=512, hidden_dim=512, n_freqs=3).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=5e-5)

sf = sorted(glob.glob("aneurysm_project/data/processed_sdf/surface/*.npz"))[0]
qf = sf.replace("surface", "space")
surf = torch.from_numpy(np.load(sf)["points"]).float().unsqueeze(0).to(device)
sp   = np.load(qf)
qry  = torch.from_numpy(sp["points"]).float().unsqueeze(0).to(device)
gt   = torch.from_numpy(sp["sdfs"]).float().unsqueeze(0).to(device)

print(f"Shape: {sf.split('/')[-1]}")
print(f"GT SDF range: [{gt.min():.3f}, {gt.max():.3f}]")
print("\nOverfitting one shape, 500 steps (using mu directly, no sampling):")

for step in range(501):
    opt.zero_grad()
    # Use the posterior MEAN directly — bypass stochastic sampling.
    # If this converges, the instability was the random z noise.
    mu, logvar = model.encode(surf)
    pred = model.decode(qry, mu)
    pred_flat = pred.squeeze(-1)
    pred_c = torch.clamp(pred_flat, -0.2, 0.2)
    gt_c   = torch.clamp(gt,        -0.2, 0.2)
    recon  = torch.abs(pred_c - gt_c).mean()
    recon.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 50 == 0:
        p = pred_flat.detach()
        print(f"  step {step:3d}: recon={recon.item():.5f} | "
              f"pred mean={p.mean().item():.4f} std={p.std().item():.4f}")

print("\nIf recon dropped well below 0.05, the model learns correctly.")