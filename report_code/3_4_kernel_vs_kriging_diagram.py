"""Section 3.4, Figure M3: how kernel smoothing and kriging weight nearby drifters.

Illustration only, with synthetic drifters (no data needed).
(a) kernel smoothing: fixed Gaussian weights with a 75 km bandwidth;
(b) kriging: weights from a fitted covariance, so a tight cluster of 5 drifters shares
    its weight, and each estimate comes with an uncertainty;
(c) the distance rule: kriging fits it to the data, kernel smoothing fixes it in advance.
Output: figure_m3_kernel_vs_kriging.png. Also prints the cluster weight shares
quoted in the figure.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import os
OUT = os.environ.get("REPORT_OUT", "report_outputs") + "/"; os.makedirs(OUT, exist_ok=True)
NAVY, BLUE, RED, GREY = "#00325a", "#004da9", "#b2182b", "#9aa5b1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20})
# same synthetic drifters as the presentation diagrams
rng = np.random.default_rng(4); n = 46
x = rng.uniform(0, 4, n * 2); y = rng.uniform(0, 3, n * 2)
keep = ~((x > 3) & (y > 2)) & ~((x < 1) & (y < 1) & (rng.random(len(x)) < 0.7)); x, y = x[keep][:n], y[keep][:n]
u, v = 0.6 + 0.15 * y, 0.25 * np.sin(x * 1.3); u += rng.normal(0, .18, n); v += rng.normal(0, .18, n)
x0, y0, u0, v0 = x.copy(), y.copy(), u.copy(), v.copy()   # kernel panel uses these, before the cluster
# add a tight cluster to the left of the target to show how kriging shares weight within clusters
far = np.hypot(x - 2.05, y - 1.55) > 0.35; x, y, u, v = x[far], y[far], u[far], v[far]; n = len(x)
cx = 1.45 + rng.normal(0, .04, 5); cy = 1.95 + rng.normal(0, .04, 5)
x = np.r_[x, cx]; y = np.r_[y, cy]; u = np.r_[u, 0.75 + rng.normal(0, .05, 5)]; v = np.r_[v, rng.normal(0, .05, 5)]
clus = np.r_[np.zeros(n, bool), np.ones(5, bool)]
tx, ty, L, sig2, nug = 2.05, 1.55, 1.6, 1.0, 0.3
cov = lambda d: sig2 * np.exp(-d / L)                          # exponential covariance
D = np.hypot(x[:, None] - x[None], y[:, None] - y[None]); K = cov(D) + nug * np.eye(len(x))
k = cov(np.hypot(x - tx, y - ty)); lam = np.linalg.solve(K, k)   # kriging weights
sd = np.sqrt(max(sig2 - k @ lam, 0))
uw, vw = (lam * u).sum() / lam.sum(), (lam * v).sum() / lam.sum()
wn = np.clip(lam / lam.max(), 0, 1)

fig = plt.figure(figsize=(31, 10.5)); fig.subplots_adjust(left=.01, right=.98, top=.88, bottom=.24, wspace=.12)
# ---- panel (a): kernel smoothing, fixed 75 km Gaussian weights
axk = fig.add_subplot(1, 3, 1); axk.set_xlim(-.15, 4.15); axk.set_ylim(-.15, 3.15); axk.set_aspect("equal"); axk.axis("off")
h_bw = 0.68
dk0 = np.hypot(x0 - tx, y0 - ty); wk0 = np.exp(-dk0 ** 2 / (2 * h_bw ** 2)); wk0[dk0 > 3 * h_bw] = 0
for r, a in [(3 * h_bw, .08), (2 * h_bw, .12), (h_bw, .18)]: axk.add_patch(Circle((tx, ty), r, fc=BLUE, ec="none", alpha=a, zorder=0))
for i in range(len(x0)):
    col = BLUE if wk0[i] > 0.02 else GREY; al = 0.15 + 0.85 * wk0[i]
    axk.quiver([x0[i]], [y0[i]], [u0[i]], [v0[i]], color=col, alpha=al, scale=9, width=.004 + .006 * wk0[i])
    axk.scatter([x0[i]], [y0[i]], s=25 + 90 * wk0[i], color=col, alpha=al, zorder=3)
axk.quiver([tx], [ty], [(wk0 * u0).sum() / wk0.sum()], [(wk0 * v0).sum() / wk0.sum()], color=RED, scale=4.5, width=.02, zorder=6)
axk.plot(tx, ty, "*", color=RED, ms=30, zorder=7, mec="white")
axk.annotate("", xy=(tx + h_bw, ty - 1.05), xytext=(tx, ty - 1.05), arrowprops=dict(arrowstyle="<->", color=NAVY, lw=3))
axk.text(tx + h_bw / 2, ty - 1.1, "75 km", ha="center", va="top", fontsize=22, color=NAVY)
axk.set_title("Kernel smoothing", fontsize=30, fontweight="bold", color=NAVY, pad=14)
# ---- panel (b): kriging
ax = fig.add_subplot(1, 3, 2); ax.set_xlim(-.15, 4.15); ax.set_ylim(-.15, 3.15); ax.set_aspect("equal"); ax.axis("off")
for i in range(len(x)):
    col = "#7a0177" if clus[i] else (BLUE if wn[i] > 0.05 else GREY); al = 0.15 + 0.85 * wn[i]
    ax.quiver([x[i]], [y[i]], [u[i]], [v[i]], color=col, alpha=al, scale=9, width=.004 + .006 * wn[i])
    ax.scatter([x[i]], [y[i]], s=25 + 160 * wn[i], color=col, alpha=al, zorder=3)
ax.add_patch(plt.Circle((tx, ty), 0.28, fc="none", ec=RED, lw=2.5, ls="--", zorder=5))
ax.quiver([tx], [ty], [uw], [vw], color=RED, scale=4.5, width=.02, zorder=6)
ax.plot(tx, ty, "*", color=RED, ms=30, zorder=7, mec="white")
ax.annotate("5 clustered drifters share their weight:\n12% in total (20% with kernel smoothing)", xy=(1.45, 2.02), xytext=(0.05, 2.8), fontsize=17, color="#7a0177",
            arrowprops=dict(arrowstyle="->", color="#7a0177", lw=2))
ax.annotate("uncertainty of\nthe estimate", xy=(tx + .22, ty - .18), xytext=(2.75, 0.35), fontsize=17, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=2))
ax.set_title("Kriging", fontsize=30, fontweight="bold", color=NAVY, pad=14)

# ---- panel (c): the distance rule
ax2 = fig.add_subplot(1, 3, 3)
dd = np.linspace(0, 3, 200)
pairs_d = rng.uniform(0.05, 3, 160); pairs_s = np.clip(cov(pairs_d) / (sig2 + nug) + rng.normal(0, .12, 160), -.2, 1.1)
ax2.scatter(pairs_d, pairs_s, s=22, color=GREY, alpha=.6, label="Pairs of drifters (data)")
ax2.plot(dd, cov(dd) / (sig2 + nug), color=BLUE, lw=4.5, label="Kriging: curve fitted to the data")
ax2.plot(dd, np.exp(-dd ** 2 / (2 * .68 ** 2)), color=GREY, lw=3.5, ls="--", label="Kernel: curve chosen in advance (75 km)")
ax2.set_xlabel("Distance between two drifters"); ax2.set_ylabel("How similar their currents are", labelpad=20)
ax2.set_xticks([]); ax2.set_yticks([0, 1]); ax2.set_yticklabels(["low", "high"]); ax2.set_ylim(-.25, 1.15)
for s in ("top", "right"): ax2.spines[s].set_visible(False)
ax2.set_title("Learning the distance rule", fontsize=26, fontweight="bold", color=NAVY, pad=14)
ax2.legend(loc="upper right", frameon=False, fontsize=17)
fig.legend(handles=[Line2D([], [], marker="*", color=RED, lw=0, ms=22, label="Location we estimate"),
                    Line2D([], [], marker="o", color=BLUE, lw=0, ms=14, label="Drifter with large weight"),
                    Line2D([], [], marker="o", color=BLUE, lw=0, ms=8, alpha=.3, label="Drifter with small weight"),
                    Line2D([], [], color=RED, lw=7, label="Estimated current")],
           loc="lower left", bbox_to_anchor=(.05, .02), frameon=False, fontsize=20, ncol=2)
for a_, lab in zip([axk, ax, ax2], "abc"): a_.text(0, 1.08, f"({lab})", transform=a_.transAxes, fontsize=30, fontweight="bold", color=NAVY)
fig.savefig(OUT + "figure_m3_kernel_vs_kriging.png", dpi=130, facecolor="white")
i1=np.argmin(np.abs(np.hypot(x[~clus]-tx,y[~clus]-ty)-np.hypot(1.45-tx,1.95-ty)))
print("single pt same dist", lam[~clus][i1].round(3), np.hypot(x[~clus][i1]-tx,y[~clus][i1]-ty).round(2))
print("cluster total weight", lam[clus].sum().round(3), "max single", lam[~clus].max().round(3), "each cluster pt", lam[clus].round(3))
dk = np.hypot(x - tx, y - ty); wk = np.exp(-dk**2/(2*.68**2)); wk[dk>3*.68]=0; wk/=wk.sum()
print("cluster share kernel", wk[clus].sum().round(3), " kriging", (lam[clus].sum()/lam.sum()).round(3))
