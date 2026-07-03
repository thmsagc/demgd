"""Is DEMGD importance just inverse-KDE? And what does t-step return probability add?"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, eye
from scipy.stats import spearmanr, pearsonr

rng=np.random.default_rng(0)

def build_P(X,k,include_self=True):
    n=len(X); kk=min(k+1,n)
    t=cKDTree(X); dist,idx=t.query(X,k=kk,workers=-1)
    if kk==1: dist=dist[:,None]; idx=idx[:,None]
    rows=np.repeat(np.arange(n),kk); cols=idx.ravel(); w=np.exp(-(dist.ravel()**2))
    W=csr_matrix((w,(rows,cols)),shape=(n,n)); W=W.maximum(W.T); W.sort_indices()
    rs=np.asarray(W.sum(axis=1)).ravel(); rs[rs==0]=1
    from scipy.sparse import diags
    P=diags(1.0/rs)@W
    return W,P,rs,dist

def demgd_importance(W,rs):
    return (W.diagonal()/rs)*np.diff(W.indptr)   # exactly the code's score

# datasets with varied density
def make(name,n=3000):
    if name=='gauss': return rng.normal(0,1,(n,2))
    if name=='blobs':
        c=np.array([[-8,-8],[8,8],[0,6]]); s=np.array([0.4,2.0,5.0])
        idx=rng.integers(0,3,n); return c[idx]+rng.normal(0,1,(n,2))*s[idx][:,None]
    if name=='swiss':
        from sklearn.datasets import make_swiss_roll
        X,_=make_swiss_roll(n,noise=0.2,random_state=0); return X

for name in ['gauss','blobs','swiss']:
    X=make(name); k=8
    W,P,rs,dist=build_P(X,k)
    imp=demgd_importance(W,rs)
    # candidate simpler statistics
    inv_kde = 1.0/(np.asarray(W.sum(1)).ravel()/np.diff(W.indptr))  # 1/mean affinity = inverse KDE
    mean_knn_dist = dist[:,1:].mean(1)                               # avg distance to neighbors
    unweighted_deg = np.diff(W.indptr).astype(float)
    Pii1 = P.diagonal()                                              # 1-step return prob
    # t-step return probability (P^t)_ii for t=3,10 (the ADF/HKS-like quantity)
    P3=P@P@P; P10=P3@P3@P3@P
    Pii3=P3.diagonal(); Pii10=P10.diagonal()
    def corr(a,b):
        return spearmanr(a,b).correlation
    print(f"\n[{name}]  Spearman corr of DEMGD importance vs:")
    print(f"   inverse-KDE (1/mean affinity) : {corr(imp,inv_kde):+.3f}")
    print(f"   mean kNN distance             : {corr(imp,mean_knn_dist):+.3f}")
    print(f"   unweighted degree             : {corr(imp,unweighted_deg):+.3f}")
    print(f"   1-step return P_ii            : {corr(imp,Pii1):+.3f}")
    print(f"   3-step return (P^3)_ii        : {corr(imp,Pii3):+.3f}")
    print(f"   10-step return (P^10)_ii      : {corr(imp,Pii10):+.3f}")
    # how different is t-step from 1/density?
    print(f"   (P^10)_ii vs inverse-KDE      : {corr(Pii10,inv_kde):+.3f}   <-- if <1, t-step carries extra geometry")
