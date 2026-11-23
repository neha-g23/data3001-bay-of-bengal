"""Methods Table M1: bandwidth selection by cross-validation (same setup as
the team's earlier kernel-smoothing script).

Each drifter's mean in each 1-degree cell is placed at its mean position. Drifters
are split into 10 groups (GroupKFold); each group is held out in turn and its
points are predicted by Gaussian kernel smoothing of the remaining drifters.
Vector RMSE per season, with a drifter-level standard error, scored on the points
every bandwidth can predict. Prints the lowest-error bandwidth and the
one-standard-error choice for each season.
Output: bandwidth_cv.csv
"""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import reversal_figures as rf
from common import load, OUT

H = [25, 35, 50, 60, 75, 85, 100, 125, 150]
df = load()
pts = rf.per_drifter_means(rf.select_drifters(df, "drogued"), "season")
pts = pts[pts.season.isin(["Southwest", "Northeast"])]


def smooth(tl, tn, p, h, cut=3.0):
    d = rf.haversine_km(tl, tn, p.lat.values, p.lon.values)
    w = np.exp(-0.5 * (d / h) ** 2); w[d > cut * h] = 0; s = w.sum(1)
    with np.errstate(all="ignore"):
        return w @ p.ve.values / s, w @ p.vn.values / s


out = []
for season in ["Southwest", "Northeast"]:
    sp = pts[pts.season == season].reset_index(drop=True)
    preds = {h: np.full((len(sp), 2), np.nan) for h in H}
    for tr, te in GroupKFold(n_splits=10).split(sp, groups=sp.drifter_id):
        a, b = sp.iloc[tr], sp.iloc[te]
        for h in H:
            ve, vn = smooth(b.lat.values, b.lon.values, a, h); preds[h][te] = np.c_[ve, vn]
    common = np.all([np.isfinite(preds[h]).all(1) for h in H], 0); obs = sp[["ve", "vn"]].values
    for h in H:
        se = ((preds[h] - obs) ** 2).sum(1)[common]
        per = pd.Series(se, index=sp.drifter_id.values[common]).groupby(level=0).mean()
        m = se.mean()
        out.append({"season": season, "bandwidth_km": h, "rmse": np.sqrt(m),
                    "rmse_se": per.std(ddof=1) / np.sqrt(len(per)) / (2 * np.sqrt(m)), "n_points": int(common.sum())})
T = pd.DataFrame(out); T.round(4).to_csv(OUT + "bandwidth_cv.csv", index=False); print(T.round(4).to_string(index=False))
for season, r in T.groupby("season"):
    best = r.loc[r.rmse.idxmin()]
    print(f"{season}: lowest error at {best.bandwidth_km:.0f} km; one-SE choice {r.loc[r.rmse <= best.rmse + best.rmse_se, 'bandwidth_km'].max():.0f} km")
