"""Experimental harness: DEMGD (diffusion return-probability decimation) vs baselines."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian

rng = np.random.default_rng(0)

# ---------------- diffusion importance ----------------
def diffusion_importance(X, k):
    n=len(X); kk=min(k+1,n)
    tree=cKDTree(X, balanced_tree=False, compact_nodes=False)
    dist,idx=tree.query(X,k=kk,workers=-1)
    if kk==1: dist=dist[:,None]; idx=idx[:,None]
    rows=np.repeat(np.arange(n),kk); cols=idx.ravel(); w=np.exp(-(dist.ravel()**2))
    W=csr_matrix((w,(rows,cols)),shape=(n,n)); W=W.maximum(W.T); W.sort_indices()
    rs=np.asarray(W.sum(axis=1)).ravel(); rs[rs==0]=1.0
    imp=(W.diagonal()/rs)*np.diff(W.indptr)
    return imp, W

def _local_minima(importance, W):
    n=importance.size
    order=np.lexsort((np.arange(n),importance)); key=np.empty(n,np.int64); key[order]=np.arange(n)
    indptr,indices=W.indptr,W.indices; rows=np.repeat(np.arange(n),np.diff(indptr)); big=np.int64(n+1)
    edge_key=np.where(indices!=rows, key[indices], big)
    nbr_min=np.minimum.reduceat(edge_key, indptr[:-1])
    return key < nbr_min   # bool: strict local minima of importance

# ---------------- samplers: return indices kept ----------------
def sample_demgd(X, M, k=6):
    """Iteratively remove local minima of diffusion return-prob importance until <=M kept."""
    ids=np.arange(len(X)); cur=X
    while len(cur)>M and len(cur)>k+1:
        imp,W=diffusion_importance(cur,min(k,len(cur)-1))
        lm=_local_minima(imp,W)
        rm=np.where(lm)[0]
        if rm.size==0: break
        # cap so we don't overshoot M: drop the least-important local minima first
        if len(cur)-rm.size < M:
            keepn=len(cur)-M
            rm=rm[np.argsort(imp[rm])[:keepn]]
        mask=np.ones(len(cur),bool); mask[rm]=False; cur=cur[mask]; ids=ids[mask]
    return ids

def sample_random(X, M, seed=0):
    r=np.random.default_rng(seed); return r.choice(len(X),size=M,replace=False)

def sample_fps(X, M, seed=0):
    n=len(X); r=np.random.default_rng(seed)
    sel=np.empty(M,int); sel[0]=r.integers(n)
    d=np.linalg.norm(X-X[sel[0]],axis=1)
    for i in range(1,M):
        j=np.argmax(d); sel[i]=j
        d=np.minimum(d, np.linalg.norm(X-X[j],axis=1))
    return sel

def sample_kmeans(X, M, seed=0):
    from sklearn.cluster import KMeans
    km=KMeans(n_clusters=M,n_init=3,random_state=seed).fit(X)
    # pick the real point nearest to each centroid (coreset representatives)
    t=cKDTree(X); _,idx=t.query(km.cluster_centers_,k=1)
    return np.unique(idx)

def sample_voxel(X, M):
    """Voxel-grid downsample: choose grid so that ~M occupied cells; keep 1 point per cell."""
    lo=X.min(0); hi=X.max(0); span=np.where(hi>lo,hi-lo,1.0)
    # binary search grid resolution to hit ~M
    def occupied(res):
        keys=np.floor((X-lo)/span*res).astype(np.int64)
        return keys
    g=2
    for _ in range(40):
        keys=occupied(g); uniq=np.unique(keys,axis=0)
        if len(uniq)>=M: break
        g=int(g*1.3)+1
    keys=occupied(g)
    # keep first point in each unique cell
    order=np.lexsort(keys.T)
    ks=keys[order]; first=np.concatenate(([True], np.any(ks[1:]!=ks[:-1],axis=1)))
    return order[first]

SAMPLERS={'random':sample_random,'fps':sample_fps,'kmeans':sample_kmeans,
          'voxel':sample_voxel,'demgd':sample_demgd}

# ---------------- metrics ----------------
def chamfer(A,B):
    tA=cKDTree(A); tB=cKDTree(B)
    dAB,_=tB.query(A,k=1); dBA,_=tA.query(B,k=1)
    return float(dAB.mean()+dBA.mean())

def hausdorff(A,B):
    tA=cKDTree(A); tB=cKDTree(B)
    dAB,_=tB.query(A,k=1); dBA,_=tA.query(B,k=1)
    return float(max(dAB.max(),dBA.max()))

def laplacian_spectrum(X, k=8, m=20):
    n=len(X); kk=min(k+1,n)
    t=cKDTree(X); dist,idx=t.query(X,k=kk,workers=-1)
    if kk==1: dist=dist[:,None]; idx=idx[:,None]
    sig=np.median(dist[:,1:])+1e-9
    rows=np.repeat(np.arange(n),kk); cols=idx.ravel(); w=np.exp(-(dist.ravel()**2)/(sig**2))
    W=csr_matrix((w,(rows,cols)),shape=(n,n)); W=W.maximum(W.T)
    L=laplacian(W, normed=True)
    from scipy.sparse.linalg import eigsh
    m=min(m,n-2)
    vals=eigsh(L, k=m, which='SM', return_eigenvectors=False)
    return np.sort(vals)
