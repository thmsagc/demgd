"""Is there theoretical headroom? Test whether any CHEAP diffusion-derived per-point
score approximates the GUARANTEE-BEARING importance (ridge leverage score of the
diffusion kernel) better than trivial density does. If a cheap score tracks ridge
leverage, that is a real SOTA-capable direction (cheap surrogate + inherited bounds)."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from scipy.stats import spearmanr
rng=np.random.default_rng(0)

def make(name,n=1500):
    if name=='gauss': return rng.normal(0,1,(n,2))
    if name=='blobs':
        c=np.array([[-8,-8],[8,8],[0,6]]); s=np.array([0.4,2.0,5.0])
        idx=rng.integers(0,3,n); return c[idx]+rng.normal(0,1,(n,2))*s[idx][:,None]
    if name=='swiss':
        from sklearn.datasets import make_swiss_roll
        X,_=make_swiss_roll(n,noise=0.2,random_state=0); return X
    if name=='uniform_sq': return rng.uniform(0,1,(n,2))

def kernel(X,k=12):
    n=len(X); kk=min(k+1,n)
    t=cKDTree(X); dist,idx=t.query(X,k=kk,workers=-1)
    sig=np.median(dist[:,1:])
    rows=np.repeat(np.arange(n),kk); cols=idx.ravel(); w=np.exp(-(dist.ravel()**2)/sig**2)
    W=csr_matrix((w,(rows,cols)),shape=(n,n)); W=W.maximum(W.T)
    return W.toarray()

def ridge_leverage(K, lam):
    # tau_i = (K (K+lam I)^-1)_ii  -- guarantee-bearing importance for column/point selection
    n=K.shape[0]
    M=np.linalg.solve(K+lam*np.eye(n), K)   # (K+lamI)^-1 K, symmetric-ish
    return np.diag(K@np.linalg.inv(K+lam*np.eye(n)))

for name in ['gauss','blobs','swiss','uniform_sq']:
    X=make(name); K=kernel(X)
    n=len(X)
    d=K.sum(1)                     # weighted degree (density)
    Dm=diags(1/d); P=(Dm@K)
    # candidate cheap scores
    density=d
    inv_density=1.0/d              # == current DEMGD importance (rank)
    Pii1=np.diag(P)
    P2=P@P; P4=P2@P2; P8=P4@P4
    Pii2=np.diag(P2); Pii4=np.diag(P4); Pii8=np.diag(P8)
    # ridge leverage at a couple of scales
    lam_small=0.01*np.trace(K)/n; lam_big=0.5*np.trace(K)/n
    tau_s=ridge_leverage(K,lam_small); tau_b=ridge_leverage(K,lam_big)
    def sp(a,b): return spearmanr(a,b).correlation
    print(f"\n[{name}]  |Spearman| of cheap scores vs ridge-leverage:")
    for tgt,tname in [(tau_s,'leverage(small λ)'),(tau_b,'leverage(big λ)')]:
        print(f"  vs {tname}:")
        for s,snm in [(inv_density,'1/density (DEMGD)'),(density,'density'),
                      (Pii1,'(P^1)_ii'),(Pii2,'(P^2)_ii'),(Pii4,'(P^4)_ii'),(Pii8,'(P^8)_ii')]:
            print(f"      {snm:22s} {abs(sp(s,tgt)):.3f}")
