"""Q2: do faster currents turn more, at the same distance from the coast?

Regresses turn angle on mean speed (average of the two monsoons) with a separate
intercept for each coast-distance band, so the slope compares locations at the
same distance. Also gives the rank (Spearman) correlation within bands.
95% ranges come from whole-drifter redraws (each redraw rebuilds both maps).

Report: about +18 deg of turn per 0.1 m/s (95% range 9-24 deg); rank correlation
0.37 within distance bands (0.16-0.49).
Output: speed_turn.csv. Slow: about 2-3 s per redraw (default NB=400).
"""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import reversal_figures as rf
from common import load, OUT, SEED, coast_distance_km

NB = int(os.environ.get("NB", 400))
EDGES = [0, 50, 100, 150, 200, 300, 500, 2000]
df = load(); g = rf.Grid(0.2, df)
pts = rf.per_drifter_means(rf.select_drifters(df, "drogued"), "season")
D = coast_distance_km(g)


def fields_from(p):
    return {s: rf.smooth_field(p[p.season == s], g, rf.BANDWIDTH_SEASON_KM) for s in rf.SEASON_MONTHS}


def stats(A):
    ok = A["both"]; ang = A["angle"][ok]; spd = ((A["s1"] + A["s2"]) / 2)[ok]
    band = np.digitize(D[ok], EDGES) - 1
    rhos, ws = [], []
    for b in range(len(EDGES) - 1):
        m = band == b
        if m.sum() >= 20: rhos.append(spearmanr(spd[m], ang[m]).correlation); ws.append(m.sum())
    X = np.column_stack([spd] + [(band == b).astype(float) for b in range(len(EDGES) - 1) if (band == b).sum() > 0])
    slope = np.linalg.lstsq(X, ang, rcond=None)[0][0] * 0.1
    return {"deg_per_0.1ms_same_distance": slope, "rank_corr_within_bands": np.average(rhos, weights=ws),
            "rank_corr_all": spearmanr(spd, ang).correlation}


S = stats(rf.seasonal_angle(fields_from(pts), g))
rng = np.random.default_rng(SEED); ids = pts.drifter_id.unique(); B = []
for b in range(NB):
    pick = rng.choice(ids, len(ids), replace=True)
    pb = pd.concat([pts[pts.drifter_id == i].assign(drifter_id=f"{i}_{k}") for k, i in enumerate(pick)])
    B.append(stats(rf.seasonal_angle(fields_from(pb), g)))
B = pd.DataFrame(B)
T = pd.DataFrame({"estimate": pd.Series(S), "lower 95%": B.quantile(.025), "upper 95%": B.quantile(.975)}).round(3)
T.to_csv(OUT + "speed_turn.csv"); print(T.to_string())
