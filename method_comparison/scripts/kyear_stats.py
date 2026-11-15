import numpy as np,pandas as pd
E=pd.read_pickle("kyear.pkl"); r=lambda d,c: np.sqrt(d[c].mean())
for c in ["base","k75","centre32","actualexp"]: print(c,round(r(E,c),4),"vs base %.1f%%"%((1-r(E,c)/r(E,"base"))*100))
def boot(a,b):
    per=E.groupby("syear")[[a,b]].agg(["sum","count"]);x,y,n=per[(a,"sum")].values,per[(b,"sum")].values,per[(a,"count")].values
    g=np.random.default_rng(1);d=[]
    for _ in range(4000):
        i=g.integers(0,len(n),len(n));d.append((1-np.sqrt(y[i].sum()/n[i].sum())/np.sqrt(x[i].sum()/n[i].sum()))*100)
    return (1-r(E,b)/r(E,a))*100,np.percentile(d,[2.5,97.5])
for a,b in [("k75","actualexp"),("k75","centre32"),("centre32","actualexp")]: print(f"{b} vs {a}: %.1f%% lower err, CI %s"%(boot(a,b)[0],np.round(boot(a,b)[1],1)))
print("coverage centre32 %.0f%% actualexp %.0f%%"%(E.centre32_in.mean()*100,E.actualexp_in.mean()*100))
for s in ["Southwest","Northeast"]:
    e=E[E.season==s]; print(s,{c:round(r(e,c),4) for c in ["base","k75","centre32","actualexp"]})
f={0:"2019",1:"04,06,15,17,21",2:"10,11,14",3:"05,07,08,09,13",4:"03,12,16,18,20"}
import collections
fy={2019:0,2004:1,2006:1,2015:1,2017:1,2021:1,2010:2,2011:2,2014:2,2005:3,2007:3,2008:3,2009:3,2013:3,2003:4,2012:4,2016:4,2018:4,2020:4}
E["fold"]=E.syear.map(fy)
print(E.groupby("fold")[["base","k75","centre32","actualexp"]].mean().pow(.5).round(3).assign(years=lambda d:d.index.map(f)))
