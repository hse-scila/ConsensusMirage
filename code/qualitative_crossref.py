"""Illustrative consensus-mirage cases (label/agreement only; expert comments are
withheld from the public dataset for annotator privacy)."""
import pandas as pd, re, unicodedata
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).replace("ё", "е").replace("Ё", "Е")
    return re.sub(r"\s+", " ", s).strip().rstrip(".").lower()

M = pd.read_csv(ROOT / "outputs" / "expert_vs_model_per_statement.csv", encoding="utf-8-sig")
mat = pd.read_csv(ROOT / "data" / "model_predictions_matrix.csv", sep=";").rename(columns={"Unnamed: 0": "stmt"})
mat["key"] = mat.Statement.map(norm)
txt = dict(zip(mat.key, mat.Statement))
mir = M[(M.fr_unanimous) & (M.fr_label == "disputed") & (M.true_label.isin(["true", "false"]))
        & (M.h_agree >= 0.75) & (M.h_modal == M.true_label)].copy()
mir["statement"] = mir.key.map(txt)
print(f"mirage examples (frontier unanimous 'disputed', humans clear true/false): n={len(mir)}")
print("(expert sources/comments are withheld from the public dataset for privacy)")
mir[["statement", "true_label", "h_agree", "h_disp_rate"]].to_csv(
    ROOT / "outputs" / "qual_mirage_examples.csv", index=False, encoding="utf-8-sig")
