import pandas as pd, numpy as np
from pathlib import Path
rng = np.random.default_rng(20260602)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
t = pd.read_csv(ROOT/"data"/"model_predictions_long.csv", encoding='utf-8-sig')
val = t[t.parseable].copy()
LABS = ['true','false','disputed']

def macro_f1(sub):
    f=[]
    for L in LABS:
        tp=((sub.pred==L)&(sub.true_label==L)).sum()
        fp=((sub.pred==L)&(sub.true_label!=L)).sum()
        fn=((sub.pred!=L)&(sub.true_label==L)).sum()
        p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0
        f.append(2*p*r/(p+r) if p+r else 0.0)
    return 100*np.mean(f)

def acc(sub): return 100*(sub.pred==sub.true_label).mean()

def cluster_boot(sub, fn, B=600):
    stmts = sub.stmt.unique()
    by = {s:g for s,g in sub.groupby('stmt')}
    vals=[]
    for _ in range(B):
        samp = rng.choice(stmts, size=len(stmts), replace=True)
        rep = pd.concat([by[s] for s in samp], ignore_index=True)
        vals.append(fn(rep))
    lo,hi = np.percentile(vals,[2.5,97.5])
    return lo,hi

# ---- Table A: per-model overall (pooled all conditions) ----
print("=== TABLE A: per-model overall (pooled across prompts/langs/temps/runs) ===")
rows=[]
for m,sub in val.groupby('model_disp'):
    f=macro_f1(sub); a=acc(sub)
    flo,fhi=cluster_boot(sub,macro_f1); 
    rows.append((m, sub.model_type.iloc[0], round(a,1), round(f,1), f"[{flo:.1f}, {fhi:.1f}]", len(sub)))
A=pd.DataFrame(rows,columns=['model','type','accuracy','macroF1','F1_95CI','n_pred']).sort_values('macroF1',ascending=False)
print(A.to_string(index=False))
A.to_csv(OUT/"tableA_per_model_overall.csv",index=False)

# ---- Table B: per-model x language (pooled prompts/temps/runs) ----
print("\n=== TABLE B: macro-F1 by model x language, and RU-EN gap ===")
recs=[]
for (m,lg),sub in val.groupby(['model_disp','lang']):
    recs.append((m,lg,round(macro_f1(sub),1)))
B=pd.DataFrame(recs,columns=['model','lang','macroF1']).pivot(index='model',columns='lang',values='macroF1')
B['delta_RU_EN']=(B['RU']-B['EN']).round(1)
print(B.sort_values('RU',ascending=False).to_string())
B.to_csv(OUT/"tableB_per_model_lang.csv")

print("(cross-check against original spreadsheet omitted from public repo)")
