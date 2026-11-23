"""Q2 Table V: median turn by month (from the mean Southwest-monsoon direction), near the coast and offshore,
with the share turning more than 135 deg, and 95% CIs from 200 drifter redraws.
Uses the same monthly maps as 05_monthly_switch.py. Output: monthly_completeness.csv
"""
import numpy as np, pandas as pd
from importlib import import_module
import reversal_figures as rf
from common import load, coast_distance_km, redraw, OUT, NB, SEED
mm = import_module("05_monthly_switch")
df = load(); g = rf.Grid(0.2, df)
Wsw, Wm, pm, allid = mm.monthly_weights(df, g)
near = coast_distance_km(g) < 150
def med(cnt):
    ve0, vn0 = Wsw.field(cnt); s0 = np.hypot(ve0, vn0); out = []
    for m in mm.ORDER:
        ve, vn = Wm[m].field(cnt); s = np.hypot(ve, vn)
        ok = np.isfinite(s) & np.isfinite(s0) & (s >= .1) & (s0 >= .1); a = rf.direction_change(ve, vn, ve0, vn0)
        row = []
        for r in (near, ~near):
            k = ok & r
            row += [np.median(a[k]) if k.sum() >= 30 else np.nan, (np.cos(np.radians(g.LAT[k]))*(a[k]>135)).sum()/np.cos(np.radians(g.LAT[k])).sum()*100 if k.sum()>=30 else np.nan]
        out.append(row)
    return np.array(out)
P = med(np.ones(len(allid))); rng = np.random.default_rng(SEED)
B = np.array([med(redraw(rng, len(allid))) for _ in range(NB)]); lo, hi = np.nanpercentile(B, 2.5, 0), np.nanpercentile(B, 97.5, 0)
R = pd.DataFrame({"month": [rf.MONTH_NAMES[m-1] for m in mm.ORDER]})
for j, n in enumerate(["coast_median", "coast_gt135", "off_median", "off_gt135"]):
    R[n] = [f"{P[i,j]:.0f} ({lo[i,j]:.0f}-{hi[i,j]:.0f})" if np.isfinite(P[i,j]) else "-" for i in range(12)]
print(R.to_string(index=False)); R.to_csv(OUT + "monthly_completeness.csv", index=False)
