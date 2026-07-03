"""PIVOT 2: Does DEMGD's density equalization IMPROVE downstream manifold learning
on non-uniformly-sampled manifolds? (Coifman-Lafon: density biases the embedding.)
Metric: how faithfully the diffusion-map embedding of the subset recovers the TRUE
manifold coordinate, vs random/FPS/voxel subsampling at equal budget."""
import numpy as np, lab
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr
rng=np.random.default_rng(0)

def biased_swiss(n):
    """Swiss roll sampled with a strong density gradient along the roll parameter t."""
    # sample t with density gradient (more points at small t)
    u=rng.uniform(0,1,n)**2.2                    # skews toward 0 -> non-uniform density
    t=1.5*np.pi*(1+2*u)
    h=rng.uniform(0,21,n)
    X=np.c_[t*np.cos(t), h, t*np.sin(t)]
    return X, t, h                                # true manifold coords (t,h)

def diffusion_embed(X, k=12, dim=2):
    n=len(X); kk=min(k+1,n)
    tr=cKDTree(X); dist,idx=tr.query(X,k=kk,workers=-1)
    sig=np.median(dist[:,1:])
    rows=np.repeat(np.arange(n),kk); cols=idx.ravel(); w=np.exp(-(dist.ravel()**2)/sig**2)
    W=csr_matrix((w,(rows,cols)),shape=(n,n)); W=W.maximum(W.T)
    d=np.asarray(W.sum(1)).ravel(); d[d==0]=1
    Dm=diags(1/np.sqrt(d)); S=(Dm@W@Dm)
    if n<=2000:
        vals,vecs=np.linalg.eigh(S.toarray())
        order=np.argsort(-vals); vecs=vecs[:,order]
    else:
        vals,vecs=eigsh(S,k=dim+1,which='LM',tol=1e-3,maxiter=20000)
        order=np.argsort(-vals); vecs=vecs[:,order]
    emb=(1/np.sqrt(d))[:,None]*vecs[:,1:dim+1]   # skip trivial component
    return emb

def recovery_score(X, t_true, h_true, k=10):
    """How well does the 2D diffusion embedding linearly recover (t,h)?
    Use max |Spearman| of each embedding axis with the true t (the hard, curved coord)."""
    emb=diffusion_embed(X,k,2)
    c_t=max(abs(spearmanr(emb[:,0],t_true).correlation),
            abs(spearmanr(emb[:,1],t_true).correlation))
    return c_t

print("Manifold recovery (Spearman of diffusion embedding vs TRUE roll coordinate t)")
print("higher = the subset yields a truer manifold embedding\n")
print(f"{'budget':>7}{'full':>9}{'random':>9}{'fps':>9}{'voxel':>9}{'demgd':>9}")
for M in [1200, 800, 500]:
    X,t,h=biased_swiss(4000)
    full=recovery_score(X,t,h)
    res={}
    for name in ['random','fps','voxel','demgd']:
        if name=='random': idx=lab.sample_random(X,M,0)
        elif name=='fps': idx=lab.sample_fps(X,M,0)
        elif name=='voxel': idx=lab.sample_voxel(X,M)
        else: idx=lab.sample_demgd(X,M,8)
        res[name]=recovery_score(X[idx],t[idx],h[idx])
    print(f"{M:>7}{full:>9.3f}{res['random']:>9.3f}{res['fps']:>9.3f}{res['voxel']:>9.3f}{res['demgd']:>9.3f}")

# repeat averaged over seeds for robustness
print("\nAveraged over 5 realizations, budget=800:")
agg={k:[] for k in ['random','fps','voxel','demgd']}
for s in range(5):
    rng=np.random.default_rng(s)
    X,t,h=biased_swiss(4000)
    for name in agg:
        if name=='random': idx=lab.sample_random(X,800,s)
        elif name=='fps': idx=lab.sample_fps(X,800,s)
        elif name=='voxel': idx=lab.sample_voxel(X,800)
        else: idx=lab.sample_demgd(X,800,8)
        agg[name].append(recovery_score(X[idx],t[idx],h[idx]))
for name in agg:
    a=np.array(agg[name]); print(f"   {name:8s} {a.mean():.3f} +- {a.std():.3f}")
