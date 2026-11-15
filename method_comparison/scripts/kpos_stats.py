import sys; sys.path.insert(0,"repo"); sys.path.insert(0,".")
import numpy as np, pandas as pd, reversal_figures as rf
nu=sys.argv[1]; E=pd.read_pickle(f"kpos_{nu}.pkl")
coast=np.vstack([rf._densify(rf.INDIA_COAST),rf._densify(rf.LANKA_COAST)])
E["dc"]=rf.haversine_km(E.lat.values,E.lon.values,coast[:,1],coast[:,0]).min(1)
E["off_km"]=rf.haversine_km(E.lat.values,E.lon.values,E.clat.values,E.clon.values).diagonal() if False else np.hypot((E.lat-E.clat)*111,(E.lon-E.clon)*111*np.cos(np.radians(E.lat)))
r=lambda d,c: np.sqrt(d[c].mean())
print("n",len(E),"median offset km",E.off_km.median().round(1),"max",E.off_km.max().round(1))
for c in ["base","k75","actual","centre"]: print(c, round(r(E,c),4), "vs base %.1f%%"%((1-r(E,c)/r(E,"base"))*100))
per=E.groupby("drifter_id")[["actual","centre"]].agg(["sum","count"]); a,b,n=per[("actual","sum")].values,per[("centre","sum")].values,per[("actual","count")].values
g=np.random.default_rng(1); d=[]
for _ in range(2000):
    i=g.integers(0,len(n),len(n)); d.append((1-np.sqrt(a[i].sum()/n[i].sum())/np.sqrt(b[i].sum()/n[i].sum()))*100)
print("actual vs centre: %.1f%% lower error, 95%% CI %.1f to %.1f"%((1-r(E,"actual")/r(E,"centre"))*100,*np.percentile(d,[2.5,97.5])))
print("coverage (both comps in 95%%): actual %.0f%% centre %.0f%%"%(E.actual_in.mean()*100,E.centre_in.mean()*100))
for s in ["Southwest","Northeast"]:
    e=E[E.season==s]; print(s,{c:round(r(e,c),4) for c in ["base","k75","actual","centre"]})
E["band"]=pd.cut(E.dc,[0,100,300,3000],labels=["0-100","100-300",">300"])
print(E.groupby("band")[["base","k75","actual","centre"]].mean().pow(.5).round(4))
