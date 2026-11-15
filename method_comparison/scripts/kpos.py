import sys; sys.path.insert(0,"repo"); sys.path.insert(0,".")
import numpy as np, pandas as pd, warnings
import baseline_vs_kernel as bk
from krige_lib import xy, gp
warnings.filterwarnings("ignore")
nu=float(sys.argv[1])
pts=pd.read_pickle("seasonal_pts.pkl")
pts["clat"]=pts.lat_bin+0.5; pts["clon"]=pts.lon_bin+0.5
rng=np.random.default_rng(bk.SEED); ids=pts.drifter_id.unique(); rng.shuffle(ids)
fold={d:i%bk.N_FOLDS for i,d in enumerate(ids)}; pts=pts.assign(fold=pts.drifter_id.map(fold))
rows=[]
for k in range(bk.N_FOLDS):
    tr,te=pts[pts.fold!=k],pts[pts.fold==k].reset_index(drop=True)
    be,bn=bk.baseline_predict(tr,te); ke,kn=bk.kernel_predict(tr,te,75)
    out=te[["drifter_id","fold","season","ve","vn","lat","lon","clat","clon"]].copy()
    out["base"]=(be-te.ve)**2+(bn-te.vn)**2; out["k75"]=(ke-te.ve)**2+(kn-te.vn)**2
    for tag,(la,lo) in {"actual":("lat","lon"),"centre":("clat","clon")}.items():
        pe=np.full(len(te),np.nan); pn=pe.copy(); se=pe.copy(); sn=pe.copy()
        for s in ["Southwest","Northeast"]:
            a=tr[tr.season==s]; ti=np.where(te.season.values==s)[0]
            X=xy(a[la].values,a[lo].values); Xt=xy(te[la].values[ti],te[lo].values[ti])
            for comp,P,S in [("ve",pe,se),("vn",pn,sn)]:
                g=gp(nu).fit(X,a[comp].values); m,sd=g.predict(Xt,return_std=True); P[ti]=m; S[ti]=sd
        out[tag]=(pe-te.ve)**2+(pn-te.vn)**2
        out[tag+"_in"]=(np.abs(te.ve-pe)<=1.96*se)&(np.abs(te.vn-pn)<=1.96*sn)
    rows.append(out); print("fold",k,flush=True)
E=pd.concat(rows,ignore_index=True).dropna(subset=["base","k75"]); E.to_pickle(f"kpos_{nu}.pkl")
for c in ["base","k75","actual","centre"]: print(c, round(np.sqrt(E[c].mean()),4))
