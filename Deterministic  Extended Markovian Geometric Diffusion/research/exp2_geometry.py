"""EXP2: density-adaptive behaviour + boundary/structure preservation + cost.
DEMGD's hypothesized niche: equalize density (thin dense, keep sparse) and keep
boundaries (asymmetric neighbourhoods -> high return prob), cheaply."""
import numpy as np, time, lab
from scipy.spatial import cKDTree
rng=np.random.default_rng(0)

def knn_dist_cv(X, k=6):
    """Coefficient of variation of kth-NN distance -> uniformity (lower=more uniform)."""
    t=cKDTree(X); d,_=t.query(X,k=k+1,workers=-1)
    dk=d[:,-1]; return float(dk.std()/dk.mean())

def coverage(S, X):
    t=cKDTree(S); d,_=t.query(X,k=1); return float(d.mean()), float(d.max())

def run_samplers(X, M):
    out={}
    for name in ['random','fps','voxel','kmeans','demgd']:
        t=time.perf_counter()
        if name=='random': idx=lab.sample_random(X,M,0)
        elif name=='fps': idx=lab.sample_fps(X,M,0)
        elif name=='voxel': idx=lab.sample_voxel(X,M)
        elif name=='kmeans': idx=lab.sample_kmeans(X,M,0)
        else: idx=lab.sample_demgd(X,M,6)
        dt=time.perf_counter()-t
        out[name]=(X[idx],dt,len(idx))
    return out

print("### A. Density equalization: strongly non-uniform (Gaussian) cloud, keep 25% ###")
X=rng.normal(0,1,size=(6000,2))  # dense center, sparse tails
print(f"input  knnCV={knn_dist_cv(X):.3f}")
res=run_samplers(X, 1500)
print(f"{'method':8s}{'knnCV(unif)':>13}{'cover_mean':>12}{'cover_max':>11}{'time_s':>9}")
for name,(S,dt,m) in res.items():
    cm,cx=coverage(S,X)
    print(f"{name:8s}{knn_dist_cv(S):>13.3f}{cm:>12.3f}{cx:>11.3f}{dt:>9.3f}")

print("\n### B. Boundary preservation: filled dense disk, keep 15% ###")
# uniform-density filled disk; boundary points have asymmetric neighborhoods
ang=rng.uniform(0,2*np.pi,20000); rad=np.sqrt(rng.uniform(0,1,20000))
X=np.c_[rad*np.cos(ang), rad*np.sin(ang)]
r=np.linalg.norm(X,axis=1)
boundary=r>0.9   # true rim (outer 19% area)
def boundary_retention(idx_pts):
    # fraction of kept points that are on the rim vs baseline proportion
    rr=np.linalg.norm(idx_pts,axis=1); return float((rr>0.9).mean())
print(f"input rim fraction={boundary.mean():.3f}")
res=run_samplers(X, 3000)
print(f"{'method':8s}{'rim_frac_kept':>14}{'rim_boost':>10}{'cover_max':>11}{'time_s':>9}")
base=boundary.mean()
for name,(S,dt,m) in res.items():
    rf=boundary_retention(S); cm,cx=coverage(S,X)
    print(f"{name:8s}{rf:>14.3f}{rf/base:>10.2f}{cx:>11.3f}{dt:>9.3f}")

print("\n### C. Cost scaling: DEMGD vs kmeans (keep 20%) ###")
print(f"{'N':>8}{'demgd_s':>10}{'kmeans_s':>10}{'fps_s':>10}{'speedup_vs_km':>15}")
for N in [2000,5000,10000,20000,40000]:
    X=rng.normal(0,1,size=(N,3)); M=N//5
    t=time.perf_counter(); lab.sample_demgd(X,M,6); td=time.perf_counter()-t
    t=time.perf_counter(); lab.sample_kmeans(X,M,0); tk=time.perf_counter()-t
    t=time.perf_counter(); lab.sample_fps(X,M,0); tf=time.perf_counter()-t
    print(f"{N:>8}{td:>10.3f}{tk:>10.3f}{tf:>10.3f}{tk/td:>14.1f}x")
