# Prompt templates

The study used a **2×2 prompt family** crossing two factors, administered in **both
Russian and English** — eight prompt conditions in total. Each prompt asked the
model to take the role of a medical researcher, assign exactly one of three
categories (*true* / *false* / *disputed*) under stated rules, and justify the
verdict with authoritative sources.

| Prompt | Definition detail | Answer order |
|--------|-------------------|--------------|
| **P1** | definitions only | label first, then justification |
| **P2** | definitions only | justification first, then label |
| **P3** | definitions + worked examples | label first, then justification |
| **P4** | definitions + worked examples | justification first, then label |

Each of P1–P4 exists in a Russian (RU) and an English (EN) version, giving the
files below.

> **⚠️ ACTION REQUIRED BEFORE THE ZENODO DEPOSIT IS FINALISED.**
> The verbatim prompt text lives in the evaluation harness, not in the released
> prediction files. Paste the exact strings that were sent to the models into the
> eight files below (create them next to this README), replacing the placeholders:
>
> ```
> prompts/P1_RU.txt   prompts/P1_EN.txt
> prompts/P2_RU.txt   prompts/P2_EN.txt
> prompts/P3_RU.txt   prompts/P3_EN.txt
> prompts/P4_RU.txt   prompts/P4_EN.txt
> ```
>
> Include any system prompt, the category definitions, the worked examples used in
> P3/P4, and the exact output-format instruction, so a third party can reproduce
> the elicitation exactly. Reviewer 2 explicitly asked for these (comment R2.4).

The raw per-prompt / per-language / per-model model outputs (which these templates
were used to generate) are consolidated in `../data/model_predictions_matrix.csv`
and `../data/model_predictions_long.csv`.
