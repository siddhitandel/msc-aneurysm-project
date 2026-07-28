"""
train_cvae.py
=============
Train the conditional VAE on (surface, query, sdf, label) tuples.
Identical training recipe to the working VAE, plus the class label.

Usage:
  python scripts/train_cvae.py \
      --processed_dir aneurysm_project/data/processed_sdf \
      --out_dir aneurysm_project/models/cvae \
      --n_epochs 800
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cvae_model import AneurysmCVAE, cvae_sdf_loss
from dataset import SurfaceDataset


class SDFLabelDataset(Dataset):
    def __init__(self, processed_dir, split="train", n_surface=2048,
                 n_query=2048, seed=42, augment=False):
        self.processed_dir = Path(processed_dir)
        self.n_surface = n_surface; self.n_query = n_query
        self.augment = augment and (split == "train")
        sd = SurfaceDataset(processed_dir, split=split, n_points=n_surface, seed=seed)
        self.ids = sd.ids; self.label_map = sd.label_map

    def __len__(self): return len(self.ids)

    def _rot(self, pts):
        a = np.random.uniform(0, 2*np.pi, 3)
        cx,cy,cz = np.cos(a); sx,sy,sz = np.sin(a)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        R = (Rz@Ry@Rx).astype(np.float32); return pts@R.T, R

    def __getitem__(self, idx):
        aid = self.ids[idx]
        sd = np.load(self.processed_dir/"surface"/f"{aid}.npz")
        qd = np.load(self.processed_dir/"space"/f"{aid}.npz")
        surf, qry, sdf = sd["points"].copy(), qd["points"].copy(), qd["sdfs"].copy()
        if len(surf) > self.n_surface:
            surf = surf[np.random.choice(len(surf), self.n_surface, replace=False)]
        if len(qry) > self.n_query:
            sel = np.random.choice(len(qry), self.n_query, replace=False)
            qry, sdf = qry[sel], sdf[sel]
        if self.augment:
            surf, R = self._rot(surf); qry = qry@R.T
        return {"surface_points": torch.from_numpy(surf).float(),
                "query_points": torch.from_numpy(qry).float(),
                "sdfs": torch.from_numpy(sdf).float(),
                "label": torch.tensor(int(self.label_map.get(aid, 0)), dtype=torch.long)}


def kl_anneal(e, m, n): return m if e >= n else m*(e/n)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tr = SDFLabelDataset(args.processed_dir, "train", augment=args.augment)
    va = SDFLabelDataset(args.processed_dir, "val", augment=False)
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    vl = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Train {len(tr)} | Val {len(va)}")

    model = AneurysmCVAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
                         n_freqs=args.n_freqs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.n_epochs, eta_min=1e-6)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    best = float("inf"); hist = {"train_recon":[], "val_recon":[], "kl":[]}
    t0 = time.time()

    for epoch in range(1, args.n_epochs+1):
        kl_w = kl_anneal(epoch, args.kl_weight, args.kl_anneal_epochs)
        model.train(); tr_r = tr_k = 0.0
        for b in tl:
            surf = b["surface_points"].to(device); qry = b["query_points"].to(device)
            gt = b["sdfs"].to(device); lab = b["label"].to(device)
            opt.zero_grad()
            sdf, mu, logvar = model(surf, qry, lab)
            loss, recon, kl = cvae_sdf_loss(sdf, gt, mu, logvar, kl_weight=kl_w,
                                            clamp_dist=args.clamp_dist)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tr_r += recon.item(); tr_k += kl.item()
        tr_r /= len(tl); tr_k /= len(tl)

        model.eval(); va_r = 0.0
        with torch.no_grad():
            for b in vl:
                surf = b["surface_points"].to(device); qry = b["query_points"].to(device)
                gt = b["sdfs"].to(device); lab = b["label"].to(device)
                sdf, mu, logvar = model(surf, qry, lab)
                _, recon, _ = cvae_sdf_loss(sdf, gt, mu, logvar, kl_weight=kl_w,
                                            clamp_dist=args.clamp_dist)
                va_r += recon.item()
        va_r /= len(vl); sched.step()
        hist["train_recon"].append(tr_r); hist["val_recon"].append(va_r); hist["kl"].append(tr_k)

        if epoch % 20 == 0 or epoch == 1:
            print(f"E{epoch:4d}/{args.n_epochs} | recon {tr_r:.4f} | val {va_r:.4f} | "
                  f"kl {tr_k:.1f} | {time.time()-t0:.0f}s")
        if va_r < best:
            best = va_r
            torch.save({"epoch":epoch, "model_state_dict":model.state_dict(),
                        "val_recon":va_r, "latent_dim":args.latent_dim,
                        "hidden_dim":args.hidden_dim, "n_freqs":args.n_freqs},
                       out/"best_cvae.pt")
    json.dump(hist, open(out/"cvae_history.json","w"), indent=2)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min | best val {best:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_epochs", type=int, default=800)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--latent_dim", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--n_freqs", type=int, default=3)
    p.add_argument("--kl_weight", type=float, default=1e-5)
    p.add_argument("--kl_anneal_epochs", type=int, default=150)
    p.add_argument("--clamp_dist", type=float, default=0.2)
    p.add_argument("--augment", action="store_true")
    args = p.parse_args()
    train(args)