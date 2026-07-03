"""PIVOT: on a UNIFORM-density shape with sharp features (cube surface), does the
t-step diffusion return probability (P^t)_ii detect edges/corners that inverse-KDE
(=current DEMGD score) cannot? Feature != density here, so density-based scores must fail."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(1)

def cube_surface(n):
    """Uniformly sample points on the surface of a unit cube; label edge/corner proximity."""
    pts=[]
    for _ in range(n):
        face=rng.integers(6); u,v=rng.uniform(-1,1,2)
        if face==0: p=[ 1,u,v]
        elif face==1:p=[-1,u,v]
        elif face==2:p=[u, 1,v]
        elif face==3:p=[u,-1,v]
        elif face==4:p=[u,v, 1]
        else:        p=[u,v,-1]
        pts.append(p)
    X=np.array(pts,float)
    # feature = near a cube edge: at least two coords near +-1
    near=(np.abs(np.abs(X)-1)<0.12).sum(1)
    feature=near>=2   # edges/corners
    return X, feature

def build_P(X,k):
    n=len(X); kk=min(k+1,n)
    t=cKDTree(X); dist,idx=t.query(X,k=kk,workers=-1)
    rows=np.repeat(np.arange(n),kk); cols=idx.ravel(); w=np.exp(-(dist.ravel()**2)/np.median(dist[:,1:])**2)
    W=csr_matrix((w,(rows,cols)),shape=(n,n)); W=W.maximum(W.T); W.sort_indices()
    rs=np.asarray(W.sum(1)).ravel(); rs[rs==0]=1; P=diags(1/rs)@W
    return W,P,rs

X,feat=cube_surface(6000)
print(f"points={len(X)}  feature(edge/corner) fraction={feat.mean():.3f}")
k=10
W,P,rs=build_P(X,k)
inv_kde=(np.diff(W.indptr))/rs                     # = current DEMGD importance (rank == 1/density)
Pt={}
Pk=P.copy()
acc=P.copy()
for t in [1,2,4,8,16,32]:
    Pt_mat=P
    M=P.copy()
    R=M
    # compute P^t by repeated squaring-ish (small t, just multiply)
    Rmat=P
    for _ in range(t-1):
        Rmat=Rmat@P
    Pt[t]=Rmat.diagonal()

# For a KEEP score, high = keep (feature). Features should be scored HIGH.
# inverse-KDE: uniform density -> no signal. return prob: does it spike at edges?
print("\nAUC (score ranks feature points as important-to-keep; 0.5=no signal):")
def auc(s):
    # try both orientations, report the informative one
    a=roc_auc_score(feat,s); return max(a,1-a), a
for name,s in [('inverse-KDE (DEMGD now)',inv_kde)]+[(f'(P^{t})_ii',Pt[t]) for t in [1,2,4,8,16,32]]:
    best,raw=auc(s)
    orient='high@feature' if raw>=0.5 else 'LOW@feature'
    print(f"   {name:24s} AUC={best:.3f}  ({orient})")

# also curvature-free density baseline: mean knn distance
