"""EXP1: downstream classification retention. Train 1-NN on the reduced training
set, evaluate on a held-out test set, across reduction rates and samplers."""
import numpy as np, lab
from sklearn.datasets import load_digits, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def class_balanced(sampler):
    """Wrap a geometric sampler to run per class and merge (keeps class proportions)."""
    def f(X,y,M):
        idx_all=[]
        for c in np.unique(y):
            ci=np.where(y==c)[0]; mc=max(1,int(round(M*len(ci)/len(X))))
            mc=min(mc,len(ci))
            sub=sampler(X[ci], mc)
            idx_all.append(ci[sub])
        return np.concatenate(idx_all)
    return f

SAMP={
 'random': class_balanced(lambda X,M: lab.sample_random(X,M,0)),
 'fps':    class_balanced(lambda X,M: lab.sample_fps(X,M,0)),
 'kmeans': class_balanced(lambda X,M: lab.sample_kmeans(X,min(M,len(X)),0)),
 'demgd':  class_balanced(lambda X,M: lab.sample_demgd(X,M,6)),
}

def run(name, Xtr,ytr,Xte,yte, rates, reps=3):
    full=KNeighborsClassifier(1).fit(Xtr,ytr).score(Xte,yte)
    print(f"\n=== {name}  (train {len(Xtr)}, test {len(Xte)}, full-1NN acc={full:.4f}) ===")
    header="rate    " + "".join(f"{s:>10}" for s in SAMP)
    print(header)
    for r in rates:
        M=max(len(np.unique(ytr)), int(len(Xtr)*r))
        row=f"{r:<6.2f}  "
        for s,fn in SAMP.items():
            accs=[]
            for rep in range(reps):
                np.random.seed(rep)
                try:
                    idx=fn(Xtr,ytr,M)
                    a=KNeighborsClassifier(1).fit(Xtr[idx],ytr[idx]).score(Xte,yte)
                except Exception as e:
                    a=float('nan')
                accs.append(a)
            row+=f"{np.mean(accs):>10.4f}"
        print(row)
    print(f"(full-set accuracy = {full:.4f})")

for loader,nm in [(load_digits,'digits'),(load_breast_cancer,'breast_cancer'),(load_wine,'wine')]:
    d=loader(); X=StandardScaler().fit_transform(d.data.astype(float)); y=d.target
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    run(nm,Xtr,ytr,Xte,yte,[0.5,0.3,0.2,0.1,0.05])
