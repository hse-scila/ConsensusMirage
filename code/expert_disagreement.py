"""3.6 - human disagreement vs model behavior, using the de-identified rater CSV."""
import pandas as pd, numpy as np, re, unicodedata, difflib
from pathlib import Path
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parent.parent

def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).replace("ё", "е").replace("Ё", "Е")
    return re.sub(r"\s+", " ", s).strip().rstrip(".").lower()

RATERS = ["expert_1", "expert_2", "expert_3"] + [f"physician_{i}" for i in range(1, 7)]
C = {1: "false", 2: "disputed", 3: "true"}

# Matching the 108 final claims to the human-rated statements. False (default) allows a
# fuzzy fallback (difflib cutoff 0.85) -> n=96 matched claims, as reported in the paper.
# Set True to restrict to verbatim exact-text matches only (n~=82); the paper's §3.6
# conclusions are unchanged between the two subsets (robustness check).
VERBATIM_ONLY = False

h = pd.read_csv(ROOT / "data" / "human_expert_ratings_coded.csv")
h = h[h.statement_ru.notna()].copy()
h["key"] = h.statement_ru.map(norm)

def hstats(row):
    votes = [C[int(v)] for v in row[RATERS] if pd.notna(v) and int(v) in C]
    if len(votes) < 3:
        return None
    vc = pd.Series(votes).value_counts(); n = len(votes); p = vc / n
    return pd.Series({"h_modal": vc.index[0], "h_agree": vc.iloc[0] / n,
                      "h_disp_rate": votes.count("disputed") / n,
                      "h_entropy": -(p * np.log(p)).sum() / np.log(3)})

H = pd.concat([h[["key"]], h.apply(hstats, axis=1)], axis=1).dropna()

t = pd.read_csv(ROOT / "data" / "model_predictions_long.csv", encoding="utf-8-sig", low_memory=False)
mat = pd.read_csv(ROOT / "data" / "model_predictions_matrix.csv", sep=";").rename(columns={"Unnamed: 0": "stmt"})
mat["key"] = mat.Statement.map(norm)
val = t[t.parseable]; gm = val.groupby("stmt")
ms = pd.DataFrame({"true_label": gm.true_label.first(),
                   "m_disp_rate": gm.apply(lambda d: (d.pred == "disputed").mean()),
                   "m_err_rate": gm.apply(lambda d: (d.pred != d.true_label).mean())}).reset_index().merge(mat[["stmt", "key"]], on="stmt")
fr = val[val.model_type == "frontier"]
fml = fr.groupby(["stmt", "model_disp"]).pred.agg(lambda s: s.value_counts().index[0]).reset_index()
fu = {s: (len(set(g.pred)) == 1, g.pred.iloc[0]) for s, g in fml.groupby("stmt")}
ms["fr_unanimous"] = ms.stmt.map(lambda s: fu.get(s, (False, None))[0])
ms["fr_label"] = ms.stmt.map(lambda s: fu.get(s, (False, None))[1])

Hmap = {r["key"]: r for _, r in H.iterrows()}; hk = list(Hmap); rows = []
for _, m in ms.iterrows():
    k = m["key"]
    if k in Hmap:
        mk = k
    elif VERBATIM_ONLY:
        mk = None
    else:
        mk = (difflib.get_close_matches(k, hk, n=1, cutoff=0.85) or [None])[0]
    if mk:
        rows.append({**m.to_dict(), **{c: Hmap[mk][c] for c in ["h_modal", "h_agree", "h_disp_rate", "h_entropy"]}})
M = pd.DataFrame(rows)
print(f"3.6 matched {len(M)}/{len(ms)} | humans disputed {100*M.h_disp_rate.mean():.0f}% vs models {100*M.m_disp_rate.mean():.0f}%")
r1, _ = spearmanr(M.h_entropy, M.m_disp_rate)
clear = M[(M.h_agree >= 0.75) & (M.h_modal.isin(["true", "false"])) & (M.true_label == M.h_modal)]
print(f"  rho(human-disagree, model-disputed)={r1:.2f} | on {len(clear)} human-clear claims: model disputed {100*clear.m_disp_rate.mean():.0f}%, error {100*clear.m_err_rate.mean():.0f}%")
(ROOT / "outputs").mkdir(exist_ok=True)
M.to_csv(ROOT / "outputs" / "expert_vs_model_per_statement.csv", index=False, encoding="utf-8-sig")
