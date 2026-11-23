"""EDA Table E1: 1-degree cells kept for different minimum numbers of drogued drifters.

A drifter counts in a cell if it has at least 12 hourly records there in that monsoon.
Also prints the mean number of drifters per kept cell at the 3-drifter level
(report: 10.8 in the Southwest monsoon and 5.6 in the Northeast monsoon).
Output: table_e1.csv
"""
import pandas as pd
import reversal_figures as rf
from common import load, OUT

df = load()
p = rf.per_drifter_means(rf.select_drifters(df, "drogued"), "season")
p = p[p.season.isin(["Southwest", "Northeast"])]
n = p.groupby(["season", "lat_bin", "lon_bin"]).drifter_id.nunique().reset_index(name="drifters")
rows = [{"min_drifters_per_cell": k,
         **{f"{s} cells": int(((n.season == s) & (n.drifters >= k)).sum()) for s in ["Southwest", "Northeast"]}}
        for k in [1, 2, 3, 5, 10]]
T = pd.DataFrame(rows); T.to_csv(OUT + "table_e1.csv", index=False); print(T.to_string(index=False))
for s in ["Southwest", "Northeast"]:
    print(f"{s}: mean drifters per kept cell (>= 3 drifters) = {n[(n.season == s) & (n.drifters >= 3)].drifters.mean():.1f}")
