"""Q2: does the coast matter once current speed is taken into account?

Two regressions on the locations mapped in both monsoons (drogued drifters, 75 km):
  1. turn angle ~ near-coast indicator (< 150 km from India's east coast or Sri Lanka)
     + separate intercepts for seven speed groups (0-0.1, 0.1-0.15, 0.15-0.2,
     0.2-0.25, 0.25-0.3, 0.3-0.4, > 0.4 m/s): coastal minus offshore turn at similar speeds.
  2. turn angle ~ speed + near-coast indicator: degrees of turn per 0.1 m/s.
95% ranges come from whole-drifter redraws (each redraw rebuilds both maps), as in
4_2_q2_speed_turn.py. The check is repeated after removing offshore locations within
150 km of other land, where "land" is approximated by grid points that no drifter
came within 40 km of.

Report: coastal minus offshore at similar speeds about -11 deg (95% range -35 to +13);
faster currents turn about 17 deg further per 0.1 m/s (9-21).
Output: coast_same_speed.csv. Slow: about 2-3 s per redraw (default NB=400).
"""
import os
import numpy as np, pandas as pd
import reversal_figures as rf
from common import load, OUT, SEED, coast_distance_km

NB = int(os.environ.get("NB", 400))
SPEED_EDGES = [0, .1, .15, .2, .25, .3, .4, 99]
df = load(); g = rf.Grid(0.2, df)
pts = rf.per_drifter_means(rf.select_drifters(df, "drogued"), "season")
near_all = coast_distance_km(g) < 150

# approximate other land: grid points more than 40 km from every drifter record
obs = df.iloc[::24]                                   # one record per drifter-day is enough
la, lo = g.LAT.ravel(), g.LON.ravel()
dmin = np.concatenate([rf.haversine_km(la[i:i+1000], lo[i:i+1000], obs.lat.values, obs.lon.values).min(1)
                       for i in range(0, la.size, 1000)]).reshape(g.LAT.shape)
land = dmin > 40
lp = np.column_stack([g.LAT[land], g.LON[land]])
d_land = np.concatenate([rf.haversine_km(la[i:i+1000], lo[i:i+1000], lp[:, 0], lp[:, 1]).min(1)
                         for i in range(0, la.size, 1000)]).reshape(g.LAT.shape)
near_other = (d_land < 150) & ~near_all


def fields_from(p):
    return {s: rf.smooth_field(p[p.season == s], g, rf.BANDWIDTH_SEASON_KM) for s in rf.SEASON_MONTHS}


def fit(A, drop_other=False):
    ok = (A["both"] & ~near_other) if drop_other else A["both"]
    ang = A["angle"][ok]; spd = ((A["s1"] + A["s2"]) / 2)[ok]; near = near_all[ok].astype(float)
    b = np.digitize(spd, SPEED_EDGES) - 1
    X1 = np.column_stack([near] + [(b == k).astype(float) for k in range(len(SPEED_EDGES) - 1) if (b == k).any()])
    X2 = np.column_stack([spd, near, np.ones_like(spd)])
    return {"coast_minus_offshore_same_speed_deg": np.linalg.lstsq(X1, ang, rcond=None)[0][0],
            "deg_per_0.1ms": np.linalg.lstsq(X2, ang, rcond=None)[0][0] * 0.1}


def both_fits(A):
    a, b = fit(A), fit(A, drop_other=True)
    return {**a, **{k + " (other coasts removed)": v for k, v in b.items()}}


S = both_fits(rf.seasonal_angle(fields_from(pts), g))
rng = np.random.default_rng(SEED); ids = pts.drifter_id.unique(); B = []
for _ in range(NB):
    pick = rng.choice(ids, len(ids), replace=True)
    pb = pd.concat([pts[pts.drifter_id == i].assign(drifter_id=f"{i}_{k}") for k, i in enumerate(pick)])
    B.append(both_fits(rf.seasonal_angle(fields_from(pb), g)))
B = pd.DataFrame(B)
T = pd.DataFrame({"estimate": pd.Series(S), "lower 95%": B.quantile(.025), "upper 95%": B.quantile(.975)}).round(1)
T.to_csv(OUT + "coast_same_speed.csv"); print(T.to_string())
