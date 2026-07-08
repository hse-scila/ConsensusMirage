import pandas as pd, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"
t=pd.read_csv(ROOT/"data"/"model_predictions_long.csv",encoding="utf-8-sig",low_memory=False)
val=t[t.parseable].copy(); LABS=['true','false','disputed']

print("=== overall predicted label distribution (pooled) ===")
print("  TRUTH    :", (val.drop_duplicates('stmt').true_label.value_counts(normalize=True)*100).round(1).to_dict())
print("  PRED all :", (val.pred.value_counts(normalize=True)*100).round(1).to_dict())
for mt in ['frontier','russian','medical']:
    s=val[val.model_type==mt]
    print(f"  PRED {mt:9s}:", (s.pred.value_counts(normalize=True)*100).round(1).to_dict())

def pr_rc(sub,L):
    tp=((sub.pred==L)&(sub.true_label==L)).sum(); fp=((sub.pred==L)&(sub.true_label!=L)).sum(); fn=((sub.pred!=L)&(sub.true_label==L)).sum()
    p=tp/(tp+fp) if tp+fp else np.nan; r=tp/(tp+fn) if tp+fn else np.nan
    return p,r

print("\n=== per-class precision/recall by model (pooled all conditions) ===")
rows=[]
for m,sub in val.groupby('model_disp'):
    d={'model':m,'type':sub.model_type.iloc[0]}
    for L in LABS:
        p,r=pr_rc(sub,L); d[f'{L[:4]}_P']=round(100*p,1) if p==p else None; d[f'{L[:4]}_R']=round(100*r,1) if r==r else None
    # critical error: true<->false confusion
    crit=(((sub.pred=='false')&(sub.true_label=='true'))|((sub.pred=='true')&(sub.true_label=='false'))).mean()
    d['critErr%']=round(100*crit,1)
    rows.append(d)
D=pd.DataFrame(rows).sort_values('type')
print(D.to_string(index=False))
D.to_csv(OUT/"tableC_perclass.csv",index=False)

print("\n=== key ranges for prose ===")
print(f"  false-RECALL range across models : {D['fals_R'].min():.0f}–{D['fals_R'].max():.0f}%")
print(f"  disputed-RECALL range            : {D['disp_R'].min():.0f}–{D['disp_R'].max():.0f}%")
print(f"  disputed-PRECISION range         : {D['disp_P'].min():.0f}–{D['disp_P'].max():.0f}%")
print(f"  true-RECALL range                : {D['true_R'].min():.0f}–{D['true_R'].max():.0f}%")
print(f"  critical-error range             : {D['critErr%'].min():.1f}–{D['critErr%'].max():.1f}%")
# frontier-only false recall
fr=val[val.model_type=='frontier']
print(f"  frontier false-recall            : {100*pr_rc(fr,'false')[1]:.0f}%  (miss {100-100*pr_rc(fr,'false')[1]:.0f}% of false)")
