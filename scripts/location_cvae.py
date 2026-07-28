"""
location_cvae.py
================
Location-conditioned CVAE to address the coverage limitation.

The unconditional VAE covers only ~30% of real shape variety, and higher
sampling temperature doesn't help (structural, not a sampling artefact). The
fix is to condition generation on anatomical territory, so each aneurysm
"type" can be generated explicitly rather than left to random sampling.

Anatomical territories (grouped from 21 raw locations, which are too
long-tailed to condition on individually — some have only 1 example):
  ICA        (346)  - internal carotid artery group
  MCA        (191)  - middle cerebral artery group
  ACA_ACom   (148)  - anterior cerebral / communicating group
  Posterior  ( 65)  - basilar, PICA, SCA, PCA, vertebral

Same proven-stable architecture as the rupture CVAE, but the conditioning
embedding is the 4-way territory instead of the 2-way rupture status.

Usage (train):
  python scripts/location_cvae.py train \
      --processed_dir aneurysm_project/data/processed_sdf \
      --clinical_csv  ~/MSc_Project/6678442/data/clinical.csv \
      --out_dir       aneurysm_project/models/location_cvae \
      --n_epochs 800

Usage (evaluate per-territory coverage):
  python scripts/location_cvae.py evaluate \
      --processed_dir aneurysm_project/data/processed_sdf \
      --clinical_csv  ~/MSc_Project/6678442/data/clinical.csv \
      --cvae_ckpt     aneurysm_project/models/location_cvae/best_loc_cvae.pt \
      --out_dir       aneurysm_project/results_location
"""

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import trimesh
from skimage import measure
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ae_models import PositionalEncoding
from dataset import SurfaceDataset


TERRITORIES = ["ICA", "MCA", "ACA_ACom", "Posterior"]
TERR_IDX = {t: i for i, t in enumerate(TERRITORIES)}


def territory_of(loc):
    loc = str(loc)
    if loc.startswith("ICA"): return "ICA"
    if loc.startswith("MCA"): return "MCA"
    if loc.startswith("ACom") or loc.startswith("ACA") or loc == "PeriA": return "ACA_ACom"
    if loc.startswith("BA") or loc.startswith("PICA") or loc.startswith("SCA") \
       or loc.startswith("PCA") or loc.startswith("VA"): return "Posterior"
    return None


def load_location_map(clinical_csv):
    cl = pd.read_csv(clinical_csv)
    m = {}
    for _, r in cl.iterrows():
        t = territory_of(r["location"])
        if t is not None:
            m[str(r["dataset"])] = TERR_IDX[t]
    return m


# ---- Model (4-class conditional) --------------------------------------------

class LocCVAE(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=512, n_freqs=3,
                 n_classes=4, class_embed=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.enc_emb = nn.Embedding(n_classes, class_embed)
        self.dec_emb = nn.Embedding(n_classes, class_embed)

        self.point_mlp = nn.Sequential(
            nn.Linear(3,64), nn.ReLU(True), nn.Linear(64,128), nn.ReLU(True),
            nn.Linear(128,256), nn.ReLU(True), nn.Linear(256,512), nn.ReLU(True))
        self.fc_shared = nn.Sequential(
            nn.Linear(512+class_embed,512), nn.LayerNorm(512), nn.ReLU(True))
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        # NOTE: fc_logvar is initialised AFTER the generic kaiming loop below,
        # otherwise that loop overwrites it and reintroduces posterior collapse.

        self.pos_enc = PositionalEncoding(n_freqs=n_freqs)
        coord = self.pos_enc.out_dim
        inp = coord + latent_dim + class_embed
        self.dlayers = nn.ModuleList(); d = inp
        for i in range(8):
            if i == 4: d = 512 + inp
            self.dlayers.append(nn.Linear(d, 512)); d = 512
        self.out = nn.Linear(512, 1)
        # Generic init for every Linear layer FIRST ...
        for mm in self.modules():
            if isinstance(mm, nn.Linear):
                nn.init.kaiming_normal_(mm.weight, nonlinearity="relu")
                if mm.bias is not None: nn.init.zeros_(mm.bias)
        # ... then the special cases, which MUST come after or they are clobbered.
        # Small final-layer init keeps the initial SDF field near zero.
        nn.init.normal_(self.out.weight, std=0.01); nn.init.zeros_(self.out.bias)
        # Near-deterministic latent at the start of training: sigma = exp(-5/2) ~ 0.08.
        # Without this the sampled z is pure noise and the model never converges
        # (posterior collapse).
        nn.init.zeros_(self.fc_logvar.weight); nn.init.constant_(self.fc_logvar.bias, -5.0)
        print(f"LocCVAE: {sum(p.numel() for p in self.parameters()):,} params, {n_classes} territories")

    def encode(self, pts, lab):
        B,N,_ = pts.shape
        x = pts.reshape(B*N,3)
        for l in self.point_mlp: x = l(x)
        x = x.reshape(B,N,-1).max(1).values
        x = self.fc_shared(torch.cat([x, self.enc_emb(lab)], -1))
        return self.fc_mu(x), torch.clamp(self.fc_logvar(x), -10, 10)

    def decode(self, q, z, lab):
        B,N,_ = q.shape
        pe = self.pos_enc(q)
        ze = z.unsqueeze(1).expand(B,N,self.latent_dim)
        ce = self.dec_emb(lab).unsqueeze(1).expand(B,N,-1)
        inp = torch.cat([pe,ze,ce],-1); x = inp.reshape(B*N,-1); flat = x.clone()
        for i,l in enumerate(self.dlayers):
            if i==4: x = torch.cat([x,flat],-1)
            x = F.relu(l(x), True)
        return self.out(x).reshape(B,N,1)

    def forward(self, s, q, lab):
        mu, lv = self.encode(s, lab)
        z = mu + torch.randn_like(mu)*torch.exp(0.5*lv)
        return self.decode(q, z, lab), mu, lv

    @torch.no_grad()
    def generate(self, q, lab, device=None, temp=1.0):
        if device is None: device = next(self.parameters()).device
        n = lab.shape[0]
        z = torch.randn(n, self.latent_dim, device=device)*temp
        if q.dim()==2: q = q.unsqueeze(0)
        return self.decode(q.expand(n,-1,-1), z, lab)


def loss_fn(pred, gt, mu, lv, kw=1e-5, clamp=0.2, fb=0.5):
    pred = pred.squeeze(-1) if pred.dim()==3 else pred
    recon = torch.abs(torch.clamp(pred,-clamp,clamp)-torch.clamp(gt,-clamp,clamp)).mean()
    kl = torch.clamp(-0.5*(1+lv-mu.pow(2)-lv.exp()), min=fb).sum(1).mean()
    return recon + kw*kl, recon


# ---- Dataset ----------------------------------------------------------------

class LocDataset(Dataset):
    def __init__(self, processed_dir, loc_map, split="train", n=2048, seed=42, aug=False):
        self.pd = Path(processed_dir); self.n = n; self.aug = aug and split=="train"
        ds = SurfaceDataset(processed_dir, split=split, n_points=n, seed=seed)
        # keep only shapes that have a territory label
        self.ids = [i for i in ds.ids if i in loc_map]
        self.loc_map = loc_map

    def __len__(self): return len(self.ids)
    def _rot(self,p):
        a=np.random.uniform(0,2*np.pi,3);cx,cy,cz=np.cos(a);sx,sy,sz=np.sin(a)
        R=(np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])).astype(np.float32)
        return p@R.T,R
    def __getitem__(self,i):
        aid=self.ids[i]
        sd=np.load(self.pd/"surface"/f"{aid}.npz"); qd=np.load(self.pd/"space"/f"{aid}.npz")
        s,q,sdf=sd["points"].copy(),qd["points"].copy(),qd["sdfs"].copy()
        if len(s)>self.n: s=s[np.random.choice(len(s),self.n,False)]
        if len(q)>self.n:
            sel=np.random.choice(len(q),self.n,False); q,sdf=q[sel],sdf[sel]
        if self.aug: s,R=self._rot(s); q=q@R.T
        return {"s":torch.from_numpy(s).float(),"q":torch.from_numpy(q).float(),
                "sdf":torch.from_numpy(sdf).float(),
                "lab":torch.tensor(self.loc_map[aid],dtype=torch.long)}


# ---- Train ------------------------------------------------------------------

def train(args):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loc=load_location_map(args.clinical_csv)
    tr=LocDataset(args.processed_dir,loc,"train",aug=True)
    va=LocDataset(args.processed_dir,loc,"val")
    print(f"Train {len(tr)} | Val {len(va)} (territory-labelled)")
    # class balance in train
    labs=[tr.loc_map[i] for i in tr.ids]
    print("Train territory counts:", {TERRITORIES[k]:labs.count(k) for k in range(4)})
    tl=DataLoader(tr,batch_size=args.batch_size,shuffle=True,num_workers=4,pin_memory=True)
    vl=DataLoader(va,batch_size=args.batch_size,num_workers=4,pin_memory=True)
    m=LocCVAE(n_freqs=args.n_freqs).to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=args.lr,weight_decay=1e-5)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.n_epochs,eta_min=1e-6)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); best=1e9; t0=time.time()
    for ep in range(1,args.n_epochs+1):
        kw=args.kl_weight if ep>=150 else args.kl_weight*ep/150
        m.train(); tr_r=0
        for b in tl:
            s,q,gt,lab=b["s"].to(dev),b["q"].to(dev),b["sdf"].to(dev),b["lab"].to(dev)
            opt.zero_grad(); pred,mu,lv=m(s,q,lab)
            loss,recon=loss_fn(pred,gt,mu,lv,kw=kw,clamp=args.clamp_dist)
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
            tr_r+=recon.item()
        tr_r/=len(tl)
        m.eval(); vr=0
        with torch.no_grad():
            for b in vl:
                s,q,gt,lab=b["s"].to(dev),b["q"].to(dev),b["sdf"].to(dev),b["lab"].to(dev)
                pred,mu,lv=m(s,q,lab); _,recon=loss_fn(pred,gt,mu,lv,kw=kw,clamp=args.clamp_dist)
                vr+=recon.item()
        vr/=len(vl); sch.step()
        if ep%20==0 or ep==1: print(f"E{ep:4d} recon {tr_r:.4f} val {vr:.4f} {time.time()-t0:.0f}s")
        if vr<best:
            best=vr
            torch.save({"model_state_dict":m.state_dict(),"val_recon":vr,
                        "latent_dim":512,"hidden_dim":512,"n_freqs":args.n_freqs,
                        "territories":TERRITORIES},out/"best_loc_cvae.pt")
    print(f"Done {(time.time()-t0)/60:.1f}min best {best:.4f}")


# ---- Evaluate per-territory coverage ----------------------------------------

def eval_grid(m,z,lab,res,dev,batch=65536):
    lin=np.linspace(-1,1,res);xx,yy,zz=np.meshgrid(lin,lin,lin,indexing="ij")
    pts=np.stack([xx.ravel(),yy.ravel(),zz.ravel()],-1).astype(np.float32)
    pt=torch.from_numpy(pts).to(dev);vals=[]
    with torch.no_grad():
        for i in range(0,len(pt),batch):
            c=pt[i:i+batch].unsqueeze(0)
            vals.append(m.decode(c,z if z.dim()==2 else z.unsqueeze(0),lab).squeeze().cpu().numpy())
    return np.concatenate(vals).reshape(res,res,res)

def grid_pts(g,level,n=2048):
    try: v,f,_,_=measure.marching_cubes(g,level=level)
    except: return None
    r=g.shape[0]; v=v/(r-1)*2-1; me=trimesh.Trimesh(vertices=v,faces=f,process=False)
    if len(me.faces)==0: return None
    p,_=trimesh.sample.sample_surface(me,n); return p.astype(np.float32)

def chamfer(a,b):
    ta,tb=cKDTree(a),cKDTree(b); return float(tb.query(a)[0].mean()+ta.query(b)[0].mean())

def evaluate(args):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loc=load_location_map(args.clinical_csv)
    ck=torch.load(args.cvae_ckpt,map_location=dev)
    m=LocCVAE(n_freqs=ck["n_freqs"]).to(dev); m.load_state_dict(ck["model_state_dict"]); m.eval()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    # real test shapes grouped by territory
    ds=SurfaceDataset(args.processed_dir,split="test",n_points=2048)
    real_by={t:[] for t in range(4)}
    for sid in ds.ids:
        if sid in loc:
            p=np.load(Path(args.processed_dir)/"surface"/f"{sid}.npz")["points"]
            if len(p)>2048: p=p[np.random.choice(len(p),2048,False)]
            real_by[loc[sid]].append(p.astype(np.float32))

    print("\nPer-territory coverage (conditioned generation):")
    results={}
    overall_cov=[]
    for t in range(4):
        reals=real_by[t]
        if len(reals)<3:
            print(f"  {TERRITORIES[t]}: only {len(reals)} real test shapes, skipping"); continue
        # generate n shapes OF THIS territory
        gen=[]
        for _ in range(args.n_gen):
            lab=torch.tensor([t],dtype=torch.long,device=dev)
            g=grid_pts(eval_grid(m,torch.randn(1,m.latent_dim,device=dev),lab,args.resolution,dev),args.offset)
            if g is not None: gen.append(g)
        if not gen: continue
        # coverage within this territory
        D=np.zeros((len(gen),len(reals)))
        for i,gg in enumerate(gen):
            for j,rr in enumerate(reals): D[i,j]=chamfer(gg,rr)
        cov=len(set(D.argmin(1).tolist()))/len(reals)
        mmd=float(D.min(0).mean())
        results[TERRITORIES[t]]={"COV_pct":cov*100,"MMD":mmd,"n_gen":len(gen),"n_real":len(reals)}
        overall_cov.append(cov*100)
        print(f"  {TERRITORIES[t]:10s}: COV {cov*100:.1f}% | MMD {mmd:.4f} | {len(gen)} gen vs {len(reals)} real")

    results["_mean_territory_COV"]=float(np.mean(overall_cov)) if overall_cov else 0
    results["_note"]="Compare mean territory COV against unconditional COV ~30%"
    json.dump(results,open(out/"location_coverage.json","w"),indent=2)

    # bar chart
    terrs=[t for t in TERRITORIES if t in results]
    covs=[results[t]["COV_pct"] for t in terrs]
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.bar(terrs,covs,color="#1C7293")
    ax.axhline(30,color="#C0392B",linestyle="--",label="unconditional baseline ~30%")
    ax.set_ylabel("Coverage (COV) %"); ax.set_title("Per-Territory Coverage: Location-Conditioned Generation")
    ax.legend(); ax.grid(True,alpha=0.3,axis="y")
    for i,c in enumerate(covs): ax.text(i,c+1,f"{c:.0f}%",ha="center")
    plt.tight_layout(); plt.savefig(out/"location_coverage.png",dpi=150,bbox_inches="tight"); plt.close()
    print(f"\n  Mean territory COV: {results['_mean_territory_COV']:.1f}% (vs ~30% unconditional)")
    print(f"Saved: location_coverage.json, location_coverage.png")


if __name__=="__main__":
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    t=sub.add_parser("train")
    t.add_argument("--processed_dir",type=Path,required=True)
    t.add_argument("--clinical_csv",type=Path,required=True)
    t.add_argument("--out_dir",type=Path,required=True)
    t.add_argument("--n_epochs",type=int,default=800); t.add_argument("--batch_size",type=int,default=8)
    t.add_argument("--lr",type=float,default=5e-5); t.add_argument("--n_freqs",type=int,default=3)
    t.add_argument("--kl_weight",type=float,default=1e-5); t.add_argument("--clamp_dist",type=float,default=0.2)
    e=sub.add_parser("evaluate")
    e.add_argument("--processed_dir",type=Path,required=True)
    e.add_argument("--clinical_csv",type=Path,required=True)
    e.add_argument("--cvae_ckpt",type=Path,required=True)
    e.add_argument("--out_dir",type=Path,required=True)
    e.add_argument("--n_gen",type=int,default=40); e.add_argument("--resolution",type=int,default=96)
    e.add_argument("--offset",type=float,default=0.01)
    a=ap.parse_args()
    if a.cmd=="train": train(a)
    else: evaluate(a)