import sys; sys.path.insert(0,"repo"); sys.path.insert(0,".")
import numpy as np, pandas as pd, warnings
from kload import load
import reversal_figures as rf, baseline_vs_kernel as bk
from krige_lib import xy, gp
warnings.filterwarnings("ignore")
df=rf.select_drifters(load(),"drogued"); df=df[df.season.isin(["Southwest","Northeast"])].copy()
t=pd.to_datetime(df.time); df["syear"]=t.dt.year-((df.season=="Northeast")&(t.dt.month<=2)).astype(int)
keys=["season","syear","lat_bin","lon_bin","drifter_id"]
pts=(df.groupby(keys).agg(ve=("ve","mean"),vn=("vn","mean"),lat=("lat","mean"),lon=("lon","mean"),n=("ve","size")).reset_index())
pts=pts[pts.n>=rf.MIN_OBS_PER_DRIFTER].reset_index(drop=True)
pts["clat"]=pts.lat_bin+.5; pts["clon"]=pts.lon_bin+.5
# assign season-years to 5 folds, balancing record counts
cnt=pts.groupby("syear").size().sort_values(ascending=False); load_=np.zeros(5); fy={}
for y,c in cnt.items(): k=int(np.argmin(load_)); fy[y]=k; load_[k]+=c
pts["fold"]=pts.syear.map(fy)
print("records",len(pts),"fold sizes",load_.astype(int)); print({k:sorted([y for y in fy if fy[y]==k]) for k in range(5)},flush=True)
rows=[]
for k in range(5):
    tr,te=pts[pts.fold!=k],pts[pts.fold==k].reset_index(drop=True)
    be,bn=bk.baseline_predict(tr,te); ke,kn=bk.kernel_predict(tr,te,75)
    out=te[["drifter_id","syear","season","ve","vn","lat","lon"]].copy()
    out["base"]=(be-te.ve)**2+(bn-te.vn)**2; out["k75"]=(ke-te.ve)**2+(kn-te.vn)**2
    for tag,(la,lo),nu in [("centre32",("clat","clon"),1.5),("actualexp",("lat","lon"),0.5)]:
        pe=np.full(len(te),np.nan); pn=pe.copy(); se=pe.copy(); sn=pe.copy()
        for s in ["Southwest","Northeast"]:
            a=tr[tr.season==s]; ti=np.where(te.season.values==s)[0]
            if len(ti)==0: continue
            X=xy(a[la].values,a[lo].values); Xt=xy(te[la].values[ti],te[lo].values[ti])
            for comp,P,S in [("ve",pe,se),("vn",pn,sn)]:
                g=gp(nu).fit(X,a[comp].values); m,sd=g.predict(Xt,return_std=True); P[ti]=m; S[ti]=sd
        out[tag]=(pe-te.ve)**2+(pn-te.vn)**2
        out[tag+"_in"]=(np.abs(te.ve-pe)<=1.96*se)&(np.abs(te.vn-pn)<=1.96*sn)
    rows.append(out); print("fold",k,flush=True)
E=pd.concat(rows,ignore_index=True).dropna(subset=["base","k75"]); E.to_pickle("kyear.pkl")
for c in ["base","k75","centre32","actualexp"]: print(c,round(np.sqrt(E[c].mean()),4))
