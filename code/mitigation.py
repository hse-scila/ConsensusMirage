"""§3.7 — computational mitigations on existing label data (frontier ensemble):
(1) disputed-suppression threshold rule on pooled frontier votes — recover false-recall;
(2) agreement-based selective prediction on the 3-model ensemble — agreement != accuracy (the mirage)."""
import pandas as pd, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/"outputs"
t=pd.read_csv(ROOT/"data"/"model_predictions_long.csv",encoding="utf-8-sig",low_memory=False)
val=t[t.parseable]; fr=val[val.model_type=='frontier']
LABS=['true','false','disputed']
rows=[]
for stmt,g in fr.groupby('stmt'):
    tl=g.true_label.iloc[0]
    rows.append((stmt,tl,(g.pred=='true').mean(),(g.pred=='false').mean(),(g.pred=='disputed').mean()))
S=pd.DataFrame(rows,columns=['stmt','true','sh_true','sh_false','sh_disp'])
def recall(df,L,pred):
    m=df['true']==L; return (df.loc[m,pred]==L).mean()*100 if m.sum() else np.nan

print("=== (1) Disputed-suppression threshold rule (pooled frontier votes) ===")
print("    assign 'disputed' only if its vote-share >= tau, else argmax(true,false)")
print(f"{'rule':>12} {'accuracy':>9} {'false-rec':>9} {'true-rec':>9} {'disp-rec':>9}")
S['modal']=S[['sh_true','sh_false','sh_disp']].idxmax(axis=1).str.replace('sh_','').replace({'disp':'disputed'})
print(f"{'modal(base)':>12} {(S['modal']==S['true']).mean()*100:9.1f} {recall(S,'false','modal'):9.1f} {recall(S,'true','modal'):9.1f} {recall(S,'disputed','modal'):9.1f}")
for tau in [0.9,0.7,0.6,0.5]:
    S['p']=np.where(S.sh_disp>=tau,'disputed',np.where(S.sh_true>=S.sh_false,'true','false'))
    print(f"{('tau=%.2f'%tau):>12} {(S.p==S['true']).mean()*100:9.1f} {recall(S,'false','p'):9.1f} {recall(S,'true','p'):9.1f} {recall(S,'disputed','p'):9.1f}")
S['nd']=np.where(S.sh_true>=S.sh_false,'true','false')
print(f"{'never-disp':>12} {(S.nd==S['true']).mean()*100:9.1f} {recall(S,'false','nd'):9.1f} {recall(S,'true','nd'):9.1f} {recall(S,'disputed','nd'):9.1f}")

print("\n=== (2) Agreement-based selective prediction (3-model frontier ensemble) ===")
ml=fr.groupby(['stmt','model_disp']).pred.agg(lambda s:s.value_counts().index[0]).reset_index()
rec=[]
for stmt,g in ml.groupby('stmt'):
    labs=g.pred.tolist(); vc=pd.Series(labs).value_counts(); tl=fr[fr.stmt==stmt].true_label.iloc[0]
    rec.append((vc.iloc[0]/len(labs),vc.index[0]==tl))
E=pd.DataFrame(rec,columns=['agree','correct'])
una=E[E.agree==1.0]; maj=E[(E.agree>=2/3)&(E.agree<1.0)]
print(f"  all statements        : acc={E.correct.mean()*100:.1f}%  (n={len(E)})")
print(f"  unanimous (3/3 agree) : acc={una.correct.mean()*100:.1f}%  (n={len(una)})")
print(f"  split (2/3 agree)     : acc={maj.correct.mean()*100:.1f}%  (n={len(maj)})")
print("  -> requiring unanimity does NOT raise accuracy: agreement is not confidence (the mirage)")
S.to_csv(OUT/"tableF_mitigation.csv",index=False)
print("\nwrote tableF_mitigation.csv")
