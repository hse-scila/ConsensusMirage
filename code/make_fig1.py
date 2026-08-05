"""Figure 1 of the paper: the mirage in two panels.

(a) Frontier-ensemble accuracy conditioned on inter-model agreement (Wilson 95% CIs):
    selecting for unanimity does not raise accuracy.
(b) Confusion matrix over the 90 unanimously-decided claims: expert label vs unanimous
    frontier verdict, showing the retreat of true/false claims into "disputed".

Writes outputs/fig_agreement_vs_accuracy.png.
Run:  python code/make_fig1.py
"""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "model_predictions_long.csv"

t = pd.read_csv(DATA, encoding="utf-8-sig", low_memory=False)
fr = t[t.parseable & (t.model_type == "frontier")]

# collapse each frontier model to its modal label per claim, then take the ensemble vote
ml = fr.groupby(["stmt", "model_disp"]).pred.agg(lambda s: s.value_counts().index[0]).reset_index()
rows = []
for stmt, g in ml.groupby("stmt"):
    vc = g.pred.value_counts()
    tl = fr.loc[fr.stmt == stmt, "true_label"].iloc[0]
    rows.append((stmt, tl, vc.index[0], vc.iloc[0] / len(g), vc.index[0] == tl))
R = pd.DataFrame(rows, columns=["stmt", "true", "dom", "agree", "correct"])


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)


groups = [("All claims", R), ("Unanimous\n(3/3 agree)", R[R.agree == 1.0]),
          ("Split\n(2/3 agree)", R[R.agree < 1.0])]
stats = [(lab, len(d), *wilson(int(d.correct.sum()), len(d))) for lab, d in groups]

LAB = ["true", "false", "disputed"]
una = R[R.agree == 1.0]
C = pd.crosstab(pd.Categorical(una["true"], LAB), pd.Categorical(una.dom, LAB), dropna=False)
C = C.reindex(index=LAB, columns=LAB, fill_value=0)

plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10})
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9),
                         gridspec_kw={"width_ratios": [1.0, 1.12], "wspace": .34})

# ---- (a) accuracy vs. agreement ----
ax = axes[0]
x = np.arange(len(stats))
vals = [s[2] for s in stats]
err = np.array([[s[2] - s[3] for s in stats], [s[4] - s[2] for s in stats]])
ax.bar(x, vals, width=.6, color=["#6f8fae", "#2f5d8a", "#9ab0c4"],
       edgecolor="black", linewidth=.6)
ax.errorbar(x, vals, yerr=err, fmt="none", ecolor="black", elinewidth=1.1, capsize=4)
ax.axhline(33.3, ls=":", lw=1.1, color="gray")
ax.text(2.62, 33.3, "chance", fontsize=8.5, color="gray", ha="left", va="center")
for xi, s in zip(x, stats):
    ax.text(xi, s[4] + 2.5, f"{s[2]:.1f}%", ha="center", fontsize=10)
    ax.text(xi, 3.5, f"n={s[1]}", ha="center", fontsize=8.5, color="white")
ax.set_xticks(x)
ax.set_xticklabels([s[0] for s in stats], fontsize=9.5)
ax.set_ylabel("Accuracy of the ensemble verdict, %")
ax.set_ylim(0, 100); ax.set_xlim(-.6, 3.15)
ax.set_title("(a) Selecting for agreement does not raise accuracy",
             fontsize=10, loc="left", pad=10)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

# ---- (b) confusion matrix on unanimous claims ----
ax = axes[1]
M = C.values.astype(float)
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=M.max())
for i in range(3):
    for j in range(3):
        n = int(M[i, j])
        if n == 0:
            continue
        ax.text(j, i, str(n), ha="center", va="center", fontsize=13,
                color="white" if M[i, j] > M.max() * .55 else "#123",
                fontweight="bold" if i != j else "normal")
ax.set_xticks(range(3)); ax.set_xticklabels(LAB, fontsize=9.5)
ax.set_yticks(range(3)); ax.set_yticklabels(LAB, fontsize=9.5)
ax.set_xlabel("Unanimous frontier verdict")
ax.set_ylabel("Expert ground truth")
ax.set_title(f"(b) The {len(una)} claims decided unanimously",
             fontsize=10, loc="left", pad=10)
for i in range(3):
    ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False,
                               edgecolor="#2f5d8a", lw=1.8))
ax.add_patch(plt.Rectangle((1.5, -.5), 1, 2, fill=False, edgecolor="#b22", lw=2.2, ls="--"))
ax.annotate(f"{int(M[0,2]+M[1,2])} true/false claims\ncalled “disputed”",
            xy=(2.5, .5), xytext=(3.02, .5), fontsize=9.5, color="#b22", va="center",
            arrowprops=dict(arrowstyle="-", color="#b22", lw=1.0))
ax.set_xlim(-.5, 5.0); ax.set_frame_on(False)

plt.tight_layout()
fig.savefig(ROOT / "outputs" / "fig_agreement_vs_accuracy.png", dpi=300, bbox_inches="tight")
print(C, "\n")
for s in stats:
    print(f"{s[0]!r:28s} n={s[1]:3d}  acc={s[2]:.1f}%  95% CI [{s[3]:.1f}, {s[4]:.1f}]")
print("\nwrote outputs/fig_agreement_vs_accuracy.png")
