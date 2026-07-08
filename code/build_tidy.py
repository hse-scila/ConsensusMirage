"""Build a canonical tidy (long-format) prediction table from the 2026 matrix.
One row per (statement, model, prompt, language, temperature, run).
Resolves label hygiene: правда/ложь/спорно -> true/false/disputed; error/категория/blank -> NA.
"""
import pandas as pd, numpy as np, re, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root (script lives in analysis/)
SRC  = ROOT / "data" / "model_predictions_matrix.csv"
OUT  = ROOT / "outputs"

LAB = {'правда':'true','ложь':'false','спорно':'disputed'}
TYPE = {'openai':'frontier','Claude4.6':'frontier','deepseek':'frontier',
        'gigachat':'russian','yandexgpt':'russian',
        'II-Medical-8B':'medical','Med-Qwen2-7B':'medical','medgemma-27b-it':'medical',
        'medgemma-4b-it':'medical','Llama3-OpenBioLLM-70B':'medical'}
DISP = {'openai':'GPT-5.4','Claude4.6':'Claude Sonnet 4.6','deepseek':'DeepSeek-V3',
        'gigachat':'GigaChat','yandexgpt':'YandexGPT','II-Medical-8B':'II-Medical-8B',
        'Med-Qwen2-7B':'Med-Qwen2-7B','medgemma-27b-it':'MedGemma-27B-IT',
        'medgemma-4b-it':'MedGemma-4B-IT','Llama3-OpenBioLLM-70B':'Llama3-OpenBioLLM-70B'}

df = pd.read_csv(SRC, sep=';')
df = df.rename(columns={df.columns[0]:'stmt_idx'})
df['true_label'] = df['True_label'].map(LAB)
assert df['true_label'].notna().all(), "unmapped true labels"

pat = re.compile(r'^(.*)_prompt_(\d+)_(RU|EN)_t_([0-9.]+)_run_(\d+)$')
res_cols = [c for c in df.columns if pat.match(c)]
print(f"matrix: {df.shape[0]} statements x {len(res_cols)} result columns")

records = []
for c in res_cols:
    mod,pr,lg,tp,rn = pat.match(c).groups()
    raw = df[c].astype(str).str.strip()
    pred = raw.map(LAB)                 # valid labels -> canonical; else NaN
    parseable = raw.isin(LAB.keys())
    for i in range(len(df)):
        records.append((int(df.at[i,'stmt_idx']), df.at[i,'true_label'],
                        mod, TYPE[mod], DISP[mod], int(pr), lg, float(tp), int(rn),
                        pred.iat[i] if parseable.iat[i] else np.nan, bool(parseable.iat[i]),
                        raw.iat[i] if not parseable.iat[i] else ''))
tidy = pd.DataFrame.from_records(records, columns=[
    'stmt','true_label','model','model_type','model_disp','prompt','lang','temp','run',
    'pred','parseable','raw_if_unparseable'])

OUT.mkdir(parents=True, exist_ok=True)
tidy.to_csv(ROOT/"data"/"model_predictions_long.csv", index=False, encoding='utf-8-sig')
print("rows:", len(tidy), "-> wrote tidy_predictions.csv")
print("\nparseable overall: %.2f%%" % (100*tidy.parseable.mean()))
print("\nNon-parseable cell raw values (top):")
print(tidy.loc[~tidy.parseable,'raw_if_unparseable'].value_counts().head(8).to_string())

# Non-parseable rate by model
print("\n=== non-parseable rate by model ===")
g = tidy.groupby('model_disp').parseable.agg(['mean','size'])
g['nonparse_%'] = (100*(1-g['mean'])).round(2)
print(g[['nonparse_%','size']].sort_values('nonparse_%',ascending=False).to_string())

# by model x lang
print("\n=== non-parseable % by model x language ===")
p = (1-tidy.groupby(['model_disp','lang']).parseable.mean()).mul(100).round(1).unstack()
print(p.to_string())
