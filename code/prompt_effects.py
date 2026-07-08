"""§3.2 — prompt 2x2 effects: definition detail (P1,P2=short vs P3,P4=detailed) x
answer order (P1,P3=label-first vs P2,P4=reasoning-first); plus language. Inferential across model x lang units."""
import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"
t=pd.read_csv(ROOT/"data"/"model_predictions_long.csv",encoding="utf-8-sig",low_memory=False); val=t[t.parseable].copy()
LABS=['true','false','disputed']
def mf1(sub):
    f=[]
    for L in LABS:
        tp=((sub.pred==L)&(sub.true_label==L)).sum();fp=((sub.pred==L)&(sub.true_label!=L)).sum();fn=((sub.pred!=L)&(sub.true_label==L)).sum()
        p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; f.append(2*p*r/(p+r) if p+r else 0)
    return 100*np.mean(f)
rec=[(m,pr,lg,mf1(g)) for (m,pr,lg),g in val.groupby(['model_disp','prompt','lang'])]
P=pd.DataFrame(rec,columns=['model','prompt','lang','f1'])

print("=== §3.2 macro-F1 by prompt (averaged over models & languages) ===")
print(P.groupby('prompt').f1.mean().round(1).to_string())
print("\n=== macro-F1 by prompt x language (avg over models) ===")
print(P.pivot_table(index='prompt',columns='lang',values='f1',aggfunc='mean').round(1).to_string())

# 2x2 factor effects per (model,lang)
wide=P.pivot_table(index=['model','lang'],columns='prompt',values='f1')
wide.columns=[f'P{c}' for c in wide.columns]
wide['detail_eff']=(wide.P3+wide.P4)/2-(wide.P1+wide.P2)/2     # detailed - short
wide['order_eff']=(wide.P2+wide.P4)/2-(wide.P1+wide.P3)/2      # reasoning-first - label-first
print("\n=== 2x2 factor effects (macro-F1 points), across model x language units (n=%d) ===" % len(wide))
for eff,name in [('detail_eff','detailed examples vs short'),('order_eff','reasoning-first vs label-first')]:
    d=wide[eff].dropna(); 
    try: w,pp=wilcoxon(d)
    except Exception: pp=np.nan
    print(f"  {name:32s}: mean Δ={d.mean():+.1f}  median={d.median():+.1f}  Wilcoxon p={pp:.3f}")
# by family
print("\n=== mean macro-F1 by family x prompt ===")
fam={'GPT-5.4':'frontier','DeepSeek-V3':'frontier','Claude Sonnet 4.6':'frontier','GigaChat':'russian','YandexGPT':'russian'}
P['type']=P.model.map(lambda m: fam.get(m,'medical'))
print(P.pivot_table(index='type',columns='prompt',values='f1',aggfunc='mean').round(1).to_string())
wide.to_csv(OUT/"tableE_prompt_effects.csv")
print("\nwrote tableE_prompt_effects.csv")
