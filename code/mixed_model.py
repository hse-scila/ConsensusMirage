"""Confirmatory mixed-effects logistic models (Bayesian GLMM, statement+model random intercepts).
Model 1: P(correct).  Model 2: P(predict 'disputed').  Fixed effects: family, language, prompt."""
import pandas as pd, numpy as np
from pathlib import Path
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"
t=pd.read_csv(ROOT/"data"/"model_predictions_long.csv",encoding="utf-8-sig",low_memory=False)
d=t[t.parseable].copy()
d['correct']=(d.pred==d.true_label).astype(int)
d['is_disputed']=(d.pred=='disputed').astype(int)
# reference levels: frontier, EN, prompt 1
d['family']=pd.Categorical(d.model_type,categories=['frontier','medical','russian'])
d['language']=pd.Categorical(d.lang,categories=['EN','RU'])
d['promptf']=pd.Categorical(d.prompt.astype(int).astype(str),categories=['1','2','3','4'])
d['stmt']=d.stmt.astype(str); d['model_disp']=d.model_disp.astype(str)
print(f"N observations = {len(d):,} | statements={d.stmt.nunique()} | models={d.model_disp.nunique()}")

vc={"statement":"0 + C(stmt)","model":"0 + C(model_disp)"}
def fit_report(outcome,label):
    f=f"{outcome} ~ C(family) + C(language) + C(promptf)"
    m=BinomialBayesMixedGLM.from_formula(f, vc, d)
    r=m.fit_vb()
    print(f"\n================ {label}: {outcome} ================")
    names=r.model.exog_names
    print(f"{'effect':40s} {'OR':>7} {'95% CrI':>18}")
    for i,nm in enumerate(names):
        mean=r.fe_mean[i]; sd=r.fe_sd[i]
        OR=np.exp(mean); lo=np.exp(mean-1.96*sd); hi=np.exp(mean+1.96*sd)
        star='' if (lo<1<hi) else '  *'
        print(f"{nm:40s} {OR:7.2f}   [{lo:5.2f}, {hi:5.2f}]{star}")
    # random-effect SDs (posterior mean of vcp on log-sd scale -> exp)
    try:
        for j,vn in enumerate(r.model.vcp_names):
            print(f"  RE sd[{vn}] = {np.exp(r.vcp_mean[j]):.2f}")
    except Exception as e:
        print("  (RE sd parse skipped)")
    return r

r1=fit_report('correct','MODEL 1 — P(correct)')
r2=fit_report('is_disputed','MODEL 2 — P(predict disputed)')
print("\n* = 95% credible interval excludes OR=1 (frontier / EN / prompt-1 are reference levels)")
