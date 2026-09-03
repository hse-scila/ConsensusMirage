# Models evaluated

Ten models in three families, all the latest available versions at the time of
data collection (2026), each run at multiple decoding temperatures with 3
independent runs per temperature.

| Model | Family | Access | Decoding temperatures |
|-------|--------|--------|-----------------------|
| GPT-5.4 | frontier | API | 0.0, 0.5, 1.0, 1.3 |
| Claude Sonnet 4.6 | frontier | API | 0.0, 0.5, 1.0 |
| DeepSeek-V3 | frontier | API | 0.0, 0.5, 1.0, 1.3 |
| GigaChat | Russian-specific | API | 0.0, 0.5, 1.0, 1.3 |
| YandexGPT | Russian-specific | API | 0.0, 0.15, 0.30, 0.39 |
| II-Medical-8B | medical | local | 0.0, 0.5, 1.0, 1.3 |
| Med-Qwen2-7B | medical | local | 0.0, 0.5, 1.0, 1.3 |
| MedGemma-27B-IT | medical | local | 0.0, 0.5, 1.0, 1.3 |
| MedGemma-4B-IT | medical | local | 0.0, 0.5, 1.0, 1.3 |
| Llama3-OpenBioLLM-70B | medical | local | 0.0, 0.5, 1.0, 1.3 |

> **⚠️ ACTION REQUIRED BEFORE THE ZENODO DEPOSIT IS FINALISED.**
> Add the **exact identifiers** so the run can be reproduced precisely. For each
> API model give the exact endpoint / model-ID string and the access date (e.g.
> `gpt-5.4-2026-06-01`, provider, region); for each local model give the exact
> weights (Hugging Face repo + revision/commit hash) and the inference stack
> (e.g. vLLM/transformers version, quantisation, context length). Reviewer 2
> asked for model versions and settings (comment R2.4).

Per-model non-parseable rates and every individual judgement are in
`data/model_predictions_long.csv` (column `parseable`).
