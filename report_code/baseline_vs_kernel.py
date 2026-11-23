"""
Baseline vs kernel smoothing: is the smoothed map better, and is it more reliable?

Baseline = drifter-weighted gridded climatology: each drifter's mean in each
1-degree cell, then the mean over drifters in that cell (shown where >= 3
drifters). Smoothed = drifter-level Gaussian kernel (reversal_figures.py).

1. Cross-validation (5 folds, whole drifters held out, drogued drifters,
   Southwest and Northeast seasons pooled): predict the held-out drifters'
   cell-mean velocities with the baseline and with the kernel at several
   bandwidths. Vector RMSE on points both methods can predict. A drifter
   bootstrap of the held-out errors gives a 95% interval for the improvement.
2. Map bootstrap (500 resamples of whole drifters) for BOTH maps, with the same
   reversal definition (>90 deg, both seasons >= 0.1 m/s, area-weighted):
   interval of the Bay-wide reversal share and how far the Northeast arrows
   move between resamples.

Usage:  python baseline_vs_kernel.py --data bob_subset.nc --out figures
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reversal_figures as rf
import reversal_uncertainty as ru

BANDWIDTHS = [25, 50, 75, 100, 150, 200]
N_FOLDS = 5
N_BOOT = 500
SEED = 3001


def seasonal_points(df):
    pts = rf.per_drifter_means(rf.select_drifters(df, "drogued"), "season")
    return pts[pts.season.isin(["Southwest", "Northeast"])].reset_index(drop=True)


def baseline_predict(train, test):
    """Cell mean over training drifters (>= MIN_DRIFTERS) for each test point's cell."""
    g = train.groupby(["season", "lat_bin", "lon_bin"]).agg(ve=("ve", "mean"), vn=("vn", "mean"),
                                                           n=("drifter_id", "nunique")).reset_index()
    g = g[g.n >= rf.MIN_DRIFTERS]
    m = test.merge(g, on=["season", "lat_bin", "lon_bin"], how="left", suffixes=("", "_p"))
    return m.ve_p.values, m.vn_p.values


def kernel_predict(train, test, bw):
    pe = np.full(len(test), np.nan); pn = np.full(len(test), np.nan)
    for s in ["Southwest", "Northeast"]:
        tr, ti = train[train.season == s], np.where(test.season.values == s)[0]
        if len(ti) == 0: continue
        te = test.iloc[ti]
        d = rf.haversine_km(te.lat.values, te.lon.values, tr.lat.values, tr.lon.values)
        w = np.exp(-0.5 * (d / bw) ** 2); w[d > rf.KERNEL_CUTOFF * bw] = 0
        codes, _ = pd.factorize(tr.drifter_id)
        oh = np.zeros((len(tr), codes.max() + 1)); oh[np.arange(len(tr)), codes] = 1
        nn = (((d <= rf.SUPPORT_KM).astype(float) @ oh) > 0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            e = w @ tr.ve.values / w.sum(1); n = w @ tr.vn.values / w.sum(1)
        e[nn < rf.MIN_DRIFTERS] = np.nan; n[nn < rf.MIN_DRIFTERS] = np.nan
        pe[ti], pn[ti] = e, n
    return pe, pn


def cross_validate(pts, rng):
    ids = pts.drifter_id.unique(); rng.shuffle(ids)
    fold = {d: i % N_FOLDS for i, d in enumerate(ids)}
    pts = pts.assign(fold=pts.drifter_id.map(fold))
    rows = []
    for k in range(N_FOLDS):
        tr, te = pts[pts.fold != k], pts[pts.fold == k].reset_index(drop=True)
        be, bn = baseline_predict(tr, te)
        out = te[["drifter_id", "fold", "ve", "vn"]].copy(); out["base"] = (be - te.ve) ** 2 + (bn - te.vn) ** 2
        for bw in BANDWIDTHS:
            ke, kn = kernel_predict(tr, te, bw)
            out[f"k{bw}"] = (ke - te.ve) ** 2 + (kn - te.vn) ** 2
        rows.append(out)
    E = pd.concat(rows, ignore_index=True)
    return E.dropna()                       # points every method can predict


def rmse(x): return np.sqrt(np.mean(x))


def baseline_field(pts, grid1, cnt, code):
    """Weighted (bootstrap) baseline on the 1-degree grid, per season."""
    out = {}
    for s in ["Southwest", "Northeast"]:
        p = pts[pts.season == s]; c = cnt[code[p.index]]
        keep = c > 0; p, c = p[keep], c[keep]
        g = (p.assign(w=c, wve=p.ve * c, wvn=p.vn * c)
               .groupby(["lat_bin", "lon_bin"]).agg(w=("w", "sum"), wve=("wve", "sum"), wvn=("wvn", "sum"),
                                                    n=("drifter_id", "nunique")).reset_index())
        g = g[g.n >= rf.MIN_DRIFTERS]
        ve = np.full(grid1["shape"], np.nan); vn = ve.copy()
        i = (g.lat_bin - grid1["lat0"]).astype(int).values; j = (g.lon_bin - grid1["lon0"]).astype(int).values
        ve[i, j] = (g.wve / g.w).values; vn[i, j] = (g.wvn / g.w).values
        out[s] = (ve, vn)
    return out


def make(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pts = seasonal_points(df)

    # ---------- 1. cross-validation ----------
    E = cross_validate(pts, rng)
    by_fold = E.groupby("fold")[["base"] + [f"k{b}" for b in BANDWIDTHS]].mean() ** 0.5
    cv = pd.DataFrame({"rmse": [rmse(E[c]) for c in by_fold.columns],
                       "se": by_fold.std().values / np.sqrt(N_FOLDS)}, index=by_fold.columns)
    print("Held-out vector RMSE (m/s) on", len(E), "points,", E.drifter_id.nunique(), "drifters:\n", cv.round(4))
    best = cv.drop("base").rmse.idxmin()
    se_rule = cv.drop("base")[cv.drop("base").rmse <= cv.loc[best, "rmse"] + cv.loc[best, "se"]]
    print("best", best, "; within one SE:", list(se_rule.index))
    # drifter bootstrap of the improvement at 75 km
    per = E.groupby("drifter_id")[["base", "k75"]].agg(["sum", "count"])
    sb, sk, n = per[("base", "sum")].values, per[("k75", "sum")].values, per[("base", "count")].values
    imp = []
    for _ in range(2000):
        i = rng.integers(0, len(n), len(n))
        imp.append((1 - np.sqrt(sk[i].sum() / n[i].sum()) / np.sqrt(sb[i].sum() / n[i].sum())) * 100)
    imp_est = (1 - cv.loc["k75", "rmse"] / cv.loc["base", "rmse"]) * 100
    lo, hi = np.percentile(imp, [2.5, 97.5])
    print(f"RMSE reduction at 75 km: {imp_est:.1f}% (95% interval {lo:.1f} to {hi:.1f}%)")

    # ---------- 2. map bootstrap, baseline vs smoothed ----------
    codes, ids = pd.factorize(pts.drifter_id); nd = len(ids)
    lat0, lon0 = int(rf.REGION["lat"][0]), int(rf.REGION["lon"][0])
    shape = (int(rf.REGION["lat"][1]) - lat0, int(rf.REGION["lon"][1]) - lon0)
    grid1 = {"lat0": lat0, "lon0": lon0, "shape": shape}
    LAT1 = (np.arange(shape[0]) + lat0 + 0.5)[:, None] * np.ones(shape)

    def share(f, lat):
        (a, b), (c, d) = f["Southwest"], f["Northeast"]
        s1, s2 = np.hypot(a, b), np.hypot(c, d)
        fast = np.isfinite(s1) & np.isfinite(s2) & (s1 >= rf.MIN_SPEED_DIRECTION) & (s2 >= rf.MIN_SPEED_DIRECTION)
        return rf.area_share(fast, rf.direction_change(a, b, c, d) > 90, lat), fast

    gridk = rf.Grid(ru.GRID_RES, df); W, _ = ru.build(df, "drogued", gridk)
    ones = np.ones(nd)
    fb0 = baseline_field(pts, grid1, ones, codes); fk0 = {s: W[s].field(np.ones(W[s].code.max() + 1)) for s in W}
    b_est, fastb = share(fb0, LAT1); k_est, fastk = share(fk0, gridk.LAT)
    bs, ks, bt, kt = [], [], [], []
    for _ in range(N_BOOT):
        cnt = np.bincount(rng.integers(0, nd, nd), minlength=nd).astype(float)
        fb = baseline_field(pts, grid1, cnt, codes); fk = {s: W[s].field(cnt) for s in W}
        bs.append(share(fb, LAT1)[0]); ks.append(share(fk, gridk.LAT)[0])
        for f0, f, fast, store in [(fb0, fb, fastb, bt), (fk0, fk, fastk, kt)]:
            t = rf.direction_change(*f0["Northeast"], *f["Northeast"])
            store.append(np.nanmedian(np.where(fast, t, np.nan)))
    res = pd.DataFrame({
        "map": ["Baseline (1° grid)", "Smoothed (75 km)"],
        "reversal %": [b_est, k_est],
        "lower 95%": [np.percentile(bs, 2.5), np.percentile(ks, 2.5)],
        "upper 95%": [np.percentile(bs, 97.5), np.percentile(ks, 97.5)],
        "NE arrow turn between resamples (median deg)": [np.median(bt), np.median(kt)]}).round(1)
    print(res.to_string(index=False))
    res.to_csv(os.path.join(outdir, "baseline_vs_smoothed_bootstrap.csv"), index=False)
    cv.round(4).to_csv(os.path.join(outdir, "baseline_vs_smoothed_cv.csv"))

    # ---------- figure ----------
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    fig.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.14, wspace=0.32)
    ks_ = np.array(BANDWIDTHS); kr = cv.loc[[f"k{x}" for x in BANDWIDTHS], "rmse"].values
    ke = cv.loc[[f"k{x}" for x in BANDWIDTHS], "se"].values
    a.errorbar(ks_, kr, yerr=ke, fmt="o-", color="#1f5f8b", lw=2, capsize=4, label="Kernel smoothing")
    a.axhline(cv.loc["base", "rmse"], color="#b2182b", ls="--", lw=2, label="Baseline (1° grid)")
    a.plot(75, cv.loc["k75", "rmse"], "o", ms=14, mfc="none", mec="black", mew=2)
    a.annotate("chosen: 75 km", (75, cv.loc["k75", "rmse"]), xytext=(95, cv.loc["k75", "rmse"] - 0.012),
               fontsize=11, arrowprops=dict(arrowstyle="->"))
    a.set_xlabel("Bandwidth (km)"); a.set_ylabel("Held-out error, vector RMSE (m/s)")
    a.set_title(f"(a) Prediction error on held-out drifters\nsmoothing: {imp_est:.0f}% lower error (95% CI {lo:.0f}–{hi:.0f}%)",
                fontsize=12)
    a.legend(frameon=False); a.spines[["top", "right"]].set_visible(False)

    for ax, col, lab in [(b, "reversal %", "(b) Share of Bay reversing (>90°)"),
                         (c, "NE arrow turn between resamples (median deg)", "(c) Northeast arrow change\nbetween resamples")]:
        y = [1, 0]
        cols = ["#b2182b", "#1f5f8b"]
        if col == "reversal %":
            for yi, (_, r), cl in zip(y, res.iterrows(), cols):
                ax.plot([r["lower 95%"], r["upper 95%"]], [yi, yi], color=cl, lw=4, solid_capstyle="round")
                ax.plot(r[col], yi, "o", color=cl, ms=11)
                ax.text(r["upper 95%"] + 2, yi, f"{r[col]:.0f}% ({r['lower 95%']:.0f}–{r['upper 95%']:.0f})", va="center")
            ax.set_xlim(40, 110); ax.axvline(50, color="#999", ls=":"); ax.set_xlabel("% of mapped area (95% bootstrap interval)")
        else:
            ax.barh(y, res[col], color=cols, height=0.5)
            for yi, v in zip(y, res[col]): ax.text(v + 1, yi, f"{v:.0f}°", va="center", fontsize=12)
            ax.set_xlim(0, max(res[col]) * 1.3); ax.set_xlabel("Median turn (degrees): lower = more stable")
        ax.set_yticks(y); ax.set_yticklabels(["Baseline\n(1° grid)", "Smoothed\n(75 km)"]); ax.set_ylim(-0.6, 1.6)
        ax.set_title(lab, fontsize=12); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Baseline gridded map vs kernel-smoothed map (drogued drifters, 5-fold cross-validation and 500-run drifter bootstrap)",
                 fontsize=13.5)
    fig.savefig(os.path.join(outdir, "baseline_vs_smoothed.png"), dpi=140); plt.close(fig)
    return cv, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bob_subset.nc")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    make(rf.load_observations(a.data), a.out)
