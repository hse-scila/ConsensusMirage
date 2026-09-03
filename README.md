# RuHealth — The Consensus Mirage

Data and analysis code for **"The Consensus Mirage: Inter-Model Agreement Among Large Language Models Does Not Track Medical Truth"** (Artemenko et al., under review).

> This package is the **analysis-code companion** to the RuHealth corpus release. The corpus files (`ruhealth_statements.csv`, `human_expert_ratings_coded.csv`) are the canonical, de-identified RuHealth artifacts; `model_predictions_*.csv` are this study's model outputs. Everything needed to reproduce every number, table, and figure in the paper is bundled here — `bash run_all.sh` from a fresh clone.

We evaluate 10 large language models on **RuHealth**, a corpus of expert-verified Russian-language health claims, and find that agreement *among* models does not track agreement *with* medical truth: models systematically over-use a cautious "disputed" verdict, under-detect falsehoods, gain nothing from majority-vote ensembling, and — when unanimous — are wrong roughly a third of the time, almost always by laundering a true or false claim into "disputed."

---

## The dataset

**RuHealth** is a balanced corpus of **108 short Russian-language health claims** (40 true, 40 false, 28 disputed) across five topics salient in Russian-language online discourse: vaccination, genetically modified foods, healthy lifestyle and fitness, alternative medicine, and vitamins. Claims were built through a multi-stage validation pipeline: linguistic screening for clarity and single-proposition phrasing; independent veracity assessment by six physicians; source-backed annotation by three medical experts (with screen-recorded searches); editorial refinement by a medical journalist; and final adjudication by a meta-reviewer. The **disputed** category captures partial truths, exaggerations, insufficient or contradictory evidence, and irreducible ambiguity. Full construction details are in the companion resource paper (Kolmogorova et al., in preparation).

## The evaluation

| Family | Models |
|---|---|
| Frontier | GPT-5.4, Claude Sonnet 4.6, DeepSeek-V3 |
| Russian-specific | GigaChat, YandexGPT |
| Medical-specialized | II-Medical-8B, Med-Qwen2-7B, MedGemma-27B-IT, MedGemma-4B-IT, Llama3-OpenBioLLM-70B |

Each claim was classified as *true* / *false* / *disputed* under a **2×2 prompt design** (brief vs detailed category definitions × label-first vs reasoning-first), in **both Russian and a validated English translation**, across multiple decoding temperatures and 3 stochastic runs — **~90,000 judgements** in total.

---

## Repository structure

```
data/
  ruhealth_statements.csv          108 final claims: id, topic, statement_ru, statement_en, veracity
  model_predictions_long.csv       tidy one-row-per-judgement table (analysis-ready)
  model_predictions_matrix.csv     wide matrix (one column per model×prompt×lang×temp×run)
  human_expert_ratings_coded.csv   de-identified per-rater human judgements (see below)
code/                              analysis scripts (Python)
outputs/                           generated tables (CSV) and figures (PNG)
prompts/                           prompt templates (2×2 design; see prompts/README.md)
MODELS.md                          models evaluated: access mode, versions, temperatures
requirements.txt
CITATION.cff, .zenodo.json         citation and Zenodo-archival metadata
```

### Data dictionaries

**`model_predictions_long.csv`** — columns: `stmt, true_label, model, model_type, model_disp, prompt, lang, temp, run, pred, parseable, raw_if_unparseable`. Labels are `true` / `false` / `disputed`. `parseable=False` marks non-parseable model output (excluded from metrics).

**`human_expert_ratings_coded.csv`** — one row per claim; `expert_1..3` are the three source-backed medical experts, `physician_1..6` the six physicians. **Codes:** `0` = "don't know", `1` = false, `2` = disputed, `3` = true, `99` = proposed removal. `intended_veracity` is the pre-adjudication binary intent; the final adjudicated label is `veracity` in `ruhealth_statements.csv`.

> **De-identification.** Human annotators are anonymized (`expert_1..3`, `physician_1..6`); no annotator-identifying information is included. Free-text expert comments and source links from the validation phase are **not** released here.

---

## Reproducing the analysis

```bash
pip install -r requirements.txt
bash run_all.sh                  # runs all analysis scripts below, in order
```

> **One command:** `bash run_all.sh` reproduces the entire pipeline from a fresh clone (all paths are repository-relative). Generated artifacts are written to `outputs/`. To run a single step, call it directly, e.g. `python code/core_metrics.py` — scripts share only files (each reads `data/`, `mixed_model.py` and everything except `build_tidy.py` also read the tidy table), so run `build_tidy.py` first.

### What each script does

Run order is top to bottom. `build_tidy.py` regenerates `data/model_predictions_long.csv`; the rest read `data/` and write to `outputs/`.

| Script | What it computes | Writes | In the paper |
|---|---|---|---|
| `build_tidy.py` | Builds the analysis-ready long table from the wide matrix; label hygiene (правда/ложь/спорно → true/false/disputed), flags non-parseable output and reports parse rates | `data/model_predictions_long.csv` | Preprocessing; parse-rate stats |
| `core_metrics.py` | Per-model accuracy and macro-F1 with **statement-clustered bootstrap** CIs, overall and split by language | `tableA_per_model_overall.csv`, `tableB_per_model_lang.csv` | Main per-model results table |
| `descriptives.py` | Predicted-label distribution and per-class precision / recall / F1 — quantifies the over-use of *disputed* and under-detection of *false* | `tableC_perclass.csv` | Per-class analysis; disputed-bias figures |
| `consensus_mirage.py` | Inter-model agreement → accuracy relationship (the "mirage" curve, LOWESS + Wilson CIs) and the error rate under frontier unanimity | `fig_consensus_mirage.png`, `consensus_summary.json` | Main figure + consensus-mirage section |
| `make_fig1.py` | Frontier-ensemble accuracy conditioned on agreement (all / unanimous / split, Wilson CIs) and the expert-label × unanimous-verdict confusion matrix | `fig_agreement_vs_accuracy.png` | Figure 1 |
| `stability.py` | Intra-model **Krippendorff's α** across the 3 runs and its dissociation from accuracy (stable ≠ correct) | `tableD_stability.csv` | Stability section |
| `prompt_effects.py` | 2×2 prompt-design effects: definition detail (short vs detailed) × answer order (label-first vs reasoning-first) | `tableE_prompt_effects.csv` | Prompt-sensitivity section |
| `expert_disagreement.py` | Human-rater disagreement vs model *disputed* behavior; model error on human-clear claims. Matching is verbatim + fuzzy (`VERBATIM_ONLY` flag toggles the robustness subset) | `expert_vs_model_per_statement.csv` | Human-anchored analysis |
| `qualitative_crossref.py` | Illustrative consensus-mirage cases (labels / agreement only; expert free-text withheld) | `qual_mirage_examples.csv` | Illustrative-cases table |
| `mitigation.py` | Computational mitigations on the frontier ensemble: disputed-suppression rules and selective prediction (accuracy–coverage) | `tableF_mitigation.csv` | Mitigation section |
| `mixed_model.py` | Confirmatory **Bayesian mixed-effects logistic** GLMM — P(correct) and P(predict *disputed*), with statement + model random intercepts | prints coefficients | Confirmatory model |

The qualitative and human-anchored steps use only the released label / agreement data; the experts' free-text comments and source links are withheld for annotator privacy.

## Citation

If you use RuHealth or this code, please cite:

- Artemenko E, Koltsova O, et al. The Consensus Mirage: Inter-Model Agreement Among Large Language Models Does Not Track Medical Truth. *Under review*; 2026.
- Kolmogorova P, Artemenko E, Oreshina G, Koltsova O. Towards reliable research: development and medical verification of a Health Claims Corpus (RuHealth). *Manuscript in preparation*; 2025.

## License

- **Code** (`code/`) — MIT License; see `LICENSE-CODE-MIT.txt`.
- **Data** (`data/`, `outputs/`) — Creative Commons Attribution 4.0 International (CC BY 4.0); see `LICENSE-DATA-CC-BY-4.0.txt`.

## Acknowledgments

Supported by the Basic Research Program at HSE University. Social & Cognitive Informatics Laboratory, HSE University, Saint Petersburg.
