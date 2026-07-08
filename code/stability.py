"""§3.5 — intra-model stability (Krippendorff's alpha across runs) and its dissociation from accuracy;
inter-model agreement within model families."""
import pandas as pd, numpy as np, krippendorff
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"
t=pd.read_csv(ROOT/"data"/"model_predictions_long.csv",encoding="utf-8-sig",low_memory=False)
val=t[t.parseable].copy(); LBL={'true':0,'false':1,'disputed':2}; val['code']=val.pred.map(LBL)

# intra-model alpha across the 3 runs, per (model,prompt,lang,temp) with >=2 runs
recs=[]
for (m,pr,lg,tp),g in val.groupby(['model_disp','prompt','lang','temp']):
    if g.run.nunique()<2: continue
    piv=g.pivot_table(index='run',columns='stmt',values='code',aggfunc='first').values.astype(float)
    if piv.shape[0]<2 or np.all(np.isnan(piv)): continue
    try: a=krippendorff.alpha(reliability_data=piv, level_of_measurement='nominal')
    except Exception: a=np.nan
    recs.append((m,g.model_type.iloc[0],pr,lg,tp,a))
S=pd.DataFrame(recs,columns=['model','type','prompt','lang','temp','alpha'])
intra=S.groupby(['model','type']).alpha.mean().reset_index().rename(columns={'alpha':'intra_alpha'})

A=pd.read_csv(OUT/"tableA_per_model_overall.csv")[['model','accuracy','macroF1']]
ST=intra.merge(A,on='model').sort_values('intra_alpha',ascending=False)
print("=== §3.5 intra-model stability (mean Krippendorff alpha across runs) vs accuracy ===")
print(ST.round(3).to_string(index=False))
from scipy.stats import spearmanr
rho,p=spearmanr(ST.intra_alpha,ST.macroF1)
print(f"\nSpearman(intra-alpha, macro-F1) = {rho:.2f}  p={p:.3f}  -> stability does NOT imply accuracy")
print(f"  most STABLE model: {ST.iloc[0]['model']} (alpha={ST.iloc[0]['intra_alpha']:.2f}, F1={ST.iloc[0]['macroF1']:.0f})")
hi_stable_lo_acc=ST[(ST.intra_alpha>ST.intra_alpha.median())&(ST.macroF1<ST.macroF1.median())]
print("  high-stability / low-accuracy models:", hi_stable_lo_acc.model.tolist())

# inter-model agreement within families (collapse each model to modal label per stmt, per condition)
print("\n=== inter-model agreement (Krippendorff alpha) WITHIN families ===")
modal=val.groupby(['model_disp','model_type','prompt','lang','temp','stmt']).code.agg(lambda s:s.value_counts().index[0]).reset_index()
for fam in ['frontier','russian','medical']:
    fa=[]
    sub=modal[modal.model_type==fam]
    for (pr,lg,tp),g in sub.groupby(['prompt','lang','temp']):
        piv=g.pivot_table(index='model_disp',columns='stmt',values='code',aggfunc='first').values.astype(float)
        if piv.shape[0]<2: continue
        try: a=krippendorff.alpha(reliability_data=piv,level_of_measurement='nominal')
        except Exception: a=np.nan
        if a==a: fa.append(a)
    print(f"  {fam:9s}: mean pairwise-condition alpha = {np.mean(fa):.2f}  (n_conditions={len(fa)})")
ST.to_csv(OUT/"tableD_stability.csv",index=False)
print("\nwrote tableD_stability.csv")
