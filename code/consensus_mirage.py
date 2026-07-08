"""Consensus -> accuracy analysis (the 'consensus mirage').
Leads with frontier ensemble; replicates the continuous consensus curve on all model-runs.
Version-robust (no pandas-2.x-only kwargs).
"""
import pandas as pd, numpy as np, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT/"outputs"
t = pd.read_csv(ROOT/"data"/"model_predictions_long.csv", encoding="utf-8-sig", low_memory=False)
val = t[t.parseable].copy()

def modal(s):
    return s.value_counts().index[0]

def wilson(k,n,z=1.96):
    if n==0: return (0.0,0.0)
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (100*(c-h),100*(c+h))

# ===== A. ENSEMBLE: collapse each model to its modal label per statement, then vote =====
def ensemble_report(sub, name, n_models):
    ml = sub.groupby(['stmt','model_disp']).pred.agg(modal).reset_index()
    rows=[]
    for stmt,g in ml.groupby('stmt'):
        labs=g.pred.tolist(); tl=sub[sub.stmt==stmt].true_label.iloc[0]
        vc=pd.Series(labs).value_counts()
        rows.append((stmt,tl,vc.index[0],int(vc.iloc[0]),len(labs),vc.index[0]==tl,int(vc.iloc[0])==len(labs)))
    R=pd.DataFrame(rows,columns=['stmt','true','dom','domn','k','maj_correct','unanimous'])
    una=R[R.unanimous]; uw=una[~una.maj_correct]
    print(f"\n--- {name} (n_models={n_models}) ---")
    print(f"  statements: {len(R)} | majority-vote accuracy: {100*R.maj_correct.mean():.1f}%")
    print(f"  unanimous: {len(una)} | WRONG when unanimous: {len(uw)} ({100*len(uw)/max(len(una),1):.0f}%)")
    if len(uw):
        print(f"    unanimous-but-WRONG dominant label: {uw.dom.value_counts().to_dict()}")
        print(f"    unanimous-but-WRONG true label    : {uw['true'].value_counts().to_dict()}")
    return R

fr = val[val.model_type=='frontier']
Rf  = ensemble_report(fr,  "FRONTIER ensemble (GPT-5.4 / Claude 4.6 / DeepSeek)", 3)
Rall= ensemble_report(val, "ALL-10-model ensemble", 10)

# ===== C. CONTINUOUS CONSENSUS CURVE (all model-runs pooled per statement) =====
def consensus_curve(sub, label):
    rows=[]
    for stmt,g in sub.groupby('stmt'):
        votes=g.pred; tl=g.true_label.iloc[0]; n=len(votes)
        vc=votes.value_counts()
        rows.append((stmt,tl,vc.index[0],vc.iloc[0]/n,vc.index[0]==tl,(votes==tl).mean()))
    C=pd.DataFrame(rows,columns=['stmt','true','dom','dom_freq','dom_correct','correct_freq'])
    bins=[0,0.5,0.6,0.7,0.8,0.9,1.001]
    C['bin']=pd.cut(C.dom_freq,bins=bins)
    print(f"\n--- Consensus curve: {label} ({sub.model_disp.nunique()} models pooled) ---")
    for b in C['bin'].cat.categories:
        d=C[C['bin']==b]
        if len(d)==0: continue
        k=int(d.dom_correct.sum()); n=len(d); ci=wilson(k,n)
        print(f"  dom_freq {str(b):12s}: n={n:2d}  P(consensus correct)={100*k/n:5.1f}%  95%CI[{ci[0]:.0f},{ci[1]:.0f}]")
    return C

Cfr  = consensus_curve(fr,  "frontier model-runs")
Call = consensus_curve(val, "all model-runs")

# False-consensus events (all-runs pool)
hc=Call[Call.dom_freq>=0.7]; fce=hc[~hc.dom_correct]
print(f"\n--- FALSE-CONSENSUS EVENTS (all-runs pool, dom_freq>=0.70) ---")
print(f"  high-consensus: {len(hc)} | false-consensus(wrong): {len(fce)}")
print(f"  dominant label: {fce.dom.value_counts().to_dict()} | TRUE label: {fce['true'].value_counts().to_dict()}")

# ===== FIGURE =====
rng=np.random.default_rng(1)
fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
for ax,(C,ttl) in zip(axes,[(Cfr,"Frontier model-runs"),(Call,"All model-runs")]):
    lo=lowess(C.dom_correct.astype(float),C.dom_freq,frac=0.6,return_sorted=True)
    ax.scatter(C.dom_freq,100*C.dom_correct+rng.normal(0,1.2,len(C)),s=18,alpha=.35,color='#3b6')
    ax.plot(lo[:,0],100*lo[:,1],color='#b22',lw=2.2,label='LOWESS')
    ax.axhline(50,ls=':',c='gray',lw=1)
    ax.set_xlabel("Consensus strength (dominant-label freq.)"); ax.set_title(ttl); ax.set_ylim(-6,106)
axes[0].set_ylabel("P(consensus label correct), %"); axes[0].legend(loc='lower left',fontsize=8)
plt.tight_layout(); fig.savefig(OUT/"fig_consensus_mirage.png",dpi=150); print("\nwrote fig_consensus_mirage.png")

json.dump({
 "frontier_majority_acc":float(100*Rf.maj_correct.mean()),
 "frontier_unanimous":int(Rf.unanimous.sum()),
 "frontier_unanimous_wrong":int((Rf.unanimous & ~Rf.maj_correct).sum()),
 "all10_majority_acc":float(100*Rall.maj_correct.mean()),
 "false_consensus_n":int(len(fce)),
 "false_consensus_dom_disputed":int((fce.dom=='disputed').sum()),
}, open(OUT/"consensus_summary.json","w"), indent=2)
print("wrote consensus_summary.json")
